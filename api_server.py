# api_server.py
from __future__ import annotations

import json
import os
import time
import uuid
import queue
import threading
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agents import run_multi_agent
from guardrails import (
    is_disallowed,
    safe_refusal_message,
    preprocess_user_input,
    postprocess_model_output,
)
from logging_utils import log_interaction
from config import config, RuntimeConfig

# ---------------------------------------------------------------------------
# Environment & app setup
# ---------------------------------------------------------------------------

load_dotenv()

app = FastAPI(
    title="Omni-Pro V1 API",
    version="1.0.0",
    description="HTTP API for the Omni-Pro multi-agent system.",
)

# CORS – allow your web frontends to call this API
_ALLOWED_ORIGINS = os.getenv("OMNI_ALLOWED_ORIGINS", "*")
if _ALLOWED_ORIGINS == "*" or not _ALLOWED_ORIGINS.strip():
    origins = ["*"]
else:
    origins = [o.strip() for o in _ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple API key auth (optional but recommended)
_API_KEY = os.getenv("OMNI_API_KEY")


def verify_api_key(x_omni_api_key: str = Header(default=None, alias="X-Omni-API-Key")) -> None:
    """
    If OMNI_API_KEY is set in the environment, require it on each request.
    If it's not set, the API is open (for dev only).
    """
    if _API_KEY:
        if not x_omni_api_key or x_omni_api_key != _API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OmniConfigBody(BaseModel):
    # High-level routing knobs
    runtime_mode: Optional[Literal["balanced", "turbo", "deep"]] = None
    use_judge: Optional[bool] = None
    debug: Optional[bool] = None

    # Whether to include internal traces in response
    return_traces: bool = False

    # Whether to keep <think> markers in final output
    tag_thinking: bool = True


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="omni-pro-v1", description="Logical model name, e.g. omni-pro-v1")
    messages: List[ChatMessage]

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False

    user: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    omni: OmniConfigBody = Field(default_factory=OmniConfigBody)


class ChatChoice(BaseModel):
    index: int
    finish_reason: Optional[str]
    message: Dict[str, Any]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatChoice]
    omni: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_history_and_user(
    messages: List[ChatMessage],
) -> Tuple[List[Dict[str, str]], str]:
    """
    Convert OpenAI-style messages into:
      - chat_history (all but last message),
      - user_input (content of last user message).
    """
    if not messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    last = messages[-1]
    if last.role != "user":
        raise HTTPException(
            status_code=400,
            detail="The last message must have role 'user'.",
        )

    history: List[Dict[str, str]] = []
    for m in messages[:-1]:
        history.append({"role": m.role, "content": m.content})

    return history, last.content


class _RuntimeOverride:
    """
    Context manager to temporarily override config.runtime
    for a single request.
    """

    def __init__(self, omni_cfg: OmniConfigBody):
        self._orig: Optional[RuntimeConfig] = None
        self._omni_cfg = omni_cfg

    def __enter__(self):
        self._orig = RuntimeConfig(
            mode=config.runtime.mode,
            use_judge=config.runtime.use_judge,
            debug=config.runtime.debug,
        )

        if self._omni_cfg.runtime_mode is not None:
            config.runtime.mode = self._omni_cfg.runtime_mode
        if self._omni_cfg.use_judge is not None:
            config.runtime.use_judge = self._omni_cfg.use_judge
        if self._omni_cfg.debug is not None:
            config.runtime.debug = self._omni_cfg.debug

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._orig is not None:
            config.runtime = self._orig


# ---------------------------------------------------------------------------
# Non-streaming endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def create_chat_completion(
    body: ChatCompletionRequest,
    _: None = Depends(verify_api_key),
):
    """
    Main non-streaming endpoint. Returns a single ChatCompletionResponse.
    """
    if body.stream:
        # If client passed stream=true, use streaming handler
        return _streaming_chat_completion(body)

    chat_history, user_input_raw = _split_history_and_user(body.messages)

    # Guardrails
    if is_disallowed(user_input_raw):
        answer = safe_refusal_message()
        final_answer = postprocess_model_output(answer)
        try:
            log_interaction(
                user_input=user_input_raw,
                final_answer=final_answer,
                agent_traces={},
            )
        except Exception:
            pass
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        response = ChatCompletionResponse(
            id=completion_id,
            object="chat.completion",
            created=int(time.time()),
            model=body.model,
            choices=[
                ChatChoice(
                    index=0,
                    finish_reason="stop",
                    message={"role": "assistant", "content": final_answer},
                )
            ],
            omni={"blocked": True},
        )
        return JSONResponse(status_code=200, content=response.dict())

    # Normal path
    processed_input = preprocess_user_input(user_input_raw)

    with _RuntimeOverride(body.omni):
        final_answer, agent_traces = run_multi_agent(
            processed_input,
            chat_history=chat_history,
            stream_final=False,
            stream_callback=None,
        )

    final_answer = postprocess_model_output(final_answer)

    try:
        log_interaction(
            user_input=user_input_raw,
            final_answer=final_answer,
            agent_traces=agent_traces,
        )
    except Exception:
        pass

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    omni_meta: Dict[str, Any] = {
        "runtime_mode": config.runtime.mode,
        "used_judge": "judge" in agent_traces,
    }
    if body.omni.return_traces:
        omni_meta["traces"] = agent_traces

    resp = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created,
        model=body.model,
        choices=[
            ChatChoice(
                index=0,
                finish_reason="stop",
                message={
                    "role": "assistant",
                    "content": final_answer,
                },
            )
        ],
        omni=omni_meta,
    )

    return JSONResponse(status_code=200, content=resp.dict())


# ---------------------------------------------------------------------------
# Streaming endpoint (OpenAI-style SSE via the same path)
# ---------------------------------------------------------------------------


def _streaming_chat_completion(body: ChatCompletionRequest):
    """
    Handles stream=true using Server-Sent Events:
      - Content-Type: text/event-stream
      - Each line:   data: {json}\n\n
      - Final line:  data: [DONE]\n\n
    """
    chat_history, user_input_raw = _split_history_and_user(body.messages)

    if is_disallowed(user_input_raw):
        raise HTTPException(status_code=400, detail="Request content is not allowed.")

    processed_input = preprocess_user_input(user_input_raw)

    token_queue: "queue.Queue[Optional[str]]" = queue.Queue()
    result_holder: Dict[str, Any] = {
        "final_answer": "",
        "agent_traces": {},
        "error": None,
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def on_chunk(text: str) -> None:
        """
        Called by run_multi_agent whenever a partial token is ready.
        We just push it into the queue for the StreamingResponse generator.
        """
        if not text:
            return
        token_queue.put(text)

    def worker():
        try:
          with _RuntimeOverride(body.omni):
              final_answer, agent_traces = run_multi_agent(
                  processed_input,
                  chat_history=chat_history,
                  stream_final=True,
                  stream_callback=on_chunk,
              )
          final_answer = postprocess_model_output(final_answer)
          result_holder["final_answer"] = final_answer
          result_holder["agent_traces"] = agent_traces
        except Exception as e:
          result_holder["error"] = str(e)
        finally:
          # Sentinel to end the stream
          token_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def event_generator():
        first_chunk = True
        any_chunk = False

        while True:
            chunk = token_queue.get()
            if chunk is None:
                # If no chunk was streamed but we have a final_answer,
                # send it one-shot so the client still gets a reply.
                if not any_chunk and result_holder["final_answer"]:
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": body.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": result_holder["final_answer"],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                yield "data: [DONE]\n\n"

                # Log after stream completes
                if result_holder["final_answer"]:
                    try:
                        log_interaction(
                            user_input=user_input_raw,
                            final_answer=result_holder["final_answer"],
                            agent_traces=result_holder["agent_traces"],
                        )
                    except Exception:
                        pass
                break

            any_chunk = True

            delta: Dict[str, Any] = {"content": chunk}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False

            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }

            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
