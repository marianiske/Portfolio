from agent.LLM import LLM
from eval.test_cases import BASIC_CASES

def evaluate_case(llm, case):
    answer = llm.run(case["prompt"])
    trace = llm.last_run_trace or {}

    tool_calls = trace.get("tool_calls", [])
    tool_results = trace.get("tool_results", [])
    final_answer = (trace.get("final_answer") or answer or "").strip()

    tool_names = [call.get("name", "") for call in tool_calls]

    success = True
    hallucinated = False
    reasons = []

    answer_lower = final_answer.lower()

    failure_markers = [
        "failed",
        "unable",
        "could not",
        "cannot",
        "error",
        "unsuccessful",
        "did not work",
    ]

    for expected_tool in case.get("expected_tools", []):
        if expected_tool not in tool_names:
            success = False
            reasons.append(f"Missing expected tool: {expected_tool}")

    for forbidden_tool in case.get("forbidden_tools", []):
        if forbidden_tool in tool_names:
            success = False
            reasons.append(f"Forbidden tool used: {forbidden_tool}")

    expected_args = case.get("expected_args", {})
    if expected_args:
        matching_calls = [
            call for call in tool_calls
            if call.get("name") in case.get("expected_tools", [])
        ]

        if not matching_calls:
            success = False
            reasons.append("No matching tool call found for expected_args.")
        else:
            found_expected_args = False
            for call in matching_calls:
                call_args = call.get("arguments", {})
                if all(call_args.get(k) == v for k, v in expected_args.items()):
                    found_expected_args = True
                    break

            if not found_expected_args:
                success = False
                reasons.append(
                    f"Expected arguments {expected_args}, "
                    f"but got {[call.get('arguments', {}) for call in matching_calls]}"
                )

    for string in case.get("must_contain", []):
        if string.lower() not in answer.lower():
            success = False
            reasons.append(f"Model did not mention: {string}")

    if case.get("must_load_dataset"):
        expected_dataset = case.get("expected_dataset")
        if expected_dataset is not None and llm.current_dataset_name != expected_dataset:
            success = False
            reasons.append(
                f"Expected dataset {expected_dataset}, got {llm.current_dataset_name}"
            )

        loaded_results = [
            r for r in tool_results
            if r.get("name") == "find_dataset"
            and isinstance(r.get("result"), dict)
            and r["result"].get("loaded") is True
        ]
        if not loaded_results:
            success = False
            reasons.append("No successful dataset load result found in tool_results.")

    if not case.get("expect_failure", False):
        if any(marker in answer_lower for marker in failure_markers):
            success = False
            reasons.append("Final answer contains failure wording.")

    if "loaded" in answer_lower and llm.current_dataset_name is None:
        loaded_results = [
            r for r in tool_results
            if r.get("name") == "find_dataset"
            and isinstance(r.get("result"), dict)
            and r["result"].get("loaded") is True
        ]
        if not loaded_results:
            hallucinated = True
            success = False
            reasons.append(
                "Answer claims dataset was loaded, but no load result exists."
            )

    expected_dataset = case.get("expected_dataset")
    known_datasets = ["Fraud detection", "House prices", "Productivity"]

    if expected_dataset:
        for ds in known_datasets:
            if ds != expected_dataset and ds in final_answer:
                hallucinated = True
                success = False
                reasons.append(
                    f"Answer mentions unrelated dataset '{ds}' although expected dataset is '{expected_dataset}'."
                )

    if case.get("expected_tools"):
        for expected_tool in case["expected_tools"]:
            matching_results = [r for r in tool_results if r.get("name") == expected_tool]
            if not matching_results:
                success = False
                reasons.append(f"No tool result recorded for expected tool: {expected_tool}")

    forbidden_answer_terms = case.get("forbidden_answer_terms", [])
    for term in forbidden_answer_terms:
        if term.lower() in answer_lower:
            success = False
            hallucinated = True
            reasons.append(f"Answer contains forbidden content: {term}")

    if "normalize" in case.get("expected_tools", []):
        if "normalized" not in answer_lower and "normalize" not in answer_lower:
            reasons.append("Normalize test: answer does not clearly mention normalization.")

    if "pca_feature_selection" in case.get("expected_tools", []):
        pca_results = [
            r for r in tool_results
            if r.get("name") == "pca_feature_selection"
        ]
        if not pca_results:
            success = False
            reasons.append("No PCA tool result found.")
        else:
            result = pca_results[-1].get("result", {})
            if "n_features" not in result and "explained_variance_ratio" not in result:
                success = False
                reasons.append("PCA result does not contain expected PCA fields.")

    if "scatter_plot" in case.get("expected_tools", []):
        scatter_results = [
            r for r in tool_results
            if r.get("name") == "scatter_plot"
        ]
        if not scatter_results:
            success = False
            reasons.append("No scatter_plot tool result found.")
        else:
            result = scatter_results[-1].get("result", {})
            if not result.get("plot_created", False):
                success = False
                reasons.append("Scatter plot result does not confirm plot creation.")

    if case.get("must_not_dump_raw_json", False):
        if final_answer.strip().startswith("{") or final_answer.strip().startswith("["):
            success = False
            reasons.append("Answer appears to dump raw JSON.")

    return {
        "name": case["name"],
        "success": success,
        "hallucinated": hallucinated,
        "reasons": reasons,
        "answer": answer,
        "trace": trace,
    }

def summarize_results(results):
    total = len(results)
    success_count = sum(r["success"] for r in results)
    hallucination_count = sum(r["hallucinated"] for r in results)

    return {
        "total_cases": total,
        "success_rate": success_count / total if total else 0.0,
        "hallucination_rate": hallucination_count / total if total else 0.0,
    }

def main(display=False):
    results = []
    failed_tests = []
    llm = LLM()
    for ev_case in BASIC_CASES:
        llm.reset_conversation()
        res = evaluate_case(llm, ev_case)
        results.append(res)
        if display:
            print(f"Test: {ev_case['name']}, Succesful: {res['success']}\n")
            print(f'Answer: {res["answer"]}\n')
        if not res['success']:
            print(f'Test: {ev_case["name"]} failed because: Reason: {res["reasons"]}')
            failed_tests.append({'name': ev_case['name'], 'reason': res["reasons"]})
    
    print(summarize_results(results))
    print(failed_tests)

if __name__ == '__main__':
    main()