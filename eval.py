import json
from pathlib import Path
from typing import List, Dict, Any

from agents import run_multi_agent
from guardrails import preprocess_user_input, postprocess_model_output


def load_scenarios(path: str = "tests/eval_scenarios.json") -> List[Dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_eval():
    scenarios = load_scenarios()
    results = []

    for scenario in scenarios:
        prompt = scenario["prompt"]
        goal = scenario.get("goal", "")

        print("\n=======================================")
        print(f"Scenario: {scenario['id']}")
        print(f"Prompt: {prompt}")
        print(f"Goal: {goal}")
        print("=======================================")

        processed_input = preprocess_user_input(prompt)

        # For eval we don't need chat history; pass empty list.
        answer, _traces = run_multi_agent(processed_input, chat_history=[])

        answer = postprocess_model_output(answer)

        result = {
            "id": scenario["id"],
            "prompt": prompt,
            "goal": goal,
            "answer": answer,
        }
        results.append(result)

    # Save all eval results
    out_path = Path("tests/eval_results.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Evaluation complete. Results saved to tests/eval_results.json.")


if __name__ == "__main__":
    run_eval()
