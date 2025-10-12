import time
from .predict_bert import predict_relation

def print_pretty_prediction(result: dict, test_case_num: int, expected: str, claim1: str, claim2: str):
    """Pretty-print the result of a single test case."""
    prediction = result["predicted_label"]
    confidence = result["confidence"]
    probability = result["probability"]
    status = "✅ Correct" if prediction == expected else "❌ Incorrect"

    print(f"\n{'='*70}")
    print(f"Test Case {test_case_num}")
    print(f"{'='*70}")
    print(f"Claim 1: {claim1}")
    print(f"Claim 2: {claim2}")
    print(f"\nExpected:   {expected}")
    print(f"Predicted:  {prediction}")
    print(f"Probability: {probability:.4f}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Status:     {status}")


def run_tests(model, tokenizer, device, test_cases):
    results_list = []

    for i, case in enumerate(test_cases, start=1):
        result = predict_relation(case["claim1"], case["claim2"], model, tokenizer, device)
        correct = result["predicted_label"] == case["expected"]

        results_list.append({
            "test_case": i,
            "claim1": case["claim1"],
            "claim2": case["claim2"],
            "expected": case["expected"],
            "predicted": result["predicted_label"],
            "probability": result["probability"],
            "confidence": result["confidence"],
            "correct": correct
        })

    return results_list
