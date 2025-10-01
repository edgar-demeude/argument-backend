import time
from .predict import predict_relation

def print_pretty_prediction(result: dict, test_case_num: int, expected: str, claim1: str, claim2: str, best_threshold: float):
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
    print(f"Probability: {probability:.4f} (threshold: {best_threshold:.3f})")
    print(f"Confidence: {confidence:.2%}")
    print(f"Status:     {status}")


def run_tests(model, embedding_model, processor, best_threshold, label_encoder, model_type, test_cases):
    """
    Run a list of test cases and display results.
    Each test case must be a dict with keys: 'claim1', 'claim2', 'expected'
    """
    print("\n" + "="*70)
    print("RUNNING TEST CASES")
    print("="*70)

    start_time = time.time()
    correct_predictions = 0

    for i, case in enumerate(test_cases, start=1):
        result = predict_relation(
            case["claim1"],
            case["claim2"],
            model,
            embedding_model,
            processor,
            best_threshold,
            label_encoder,
            model_type,
        )
        print_pretty_prediction(
            result,
            test_case_num=i,
            expected=case["expected"],
            claim1=case["claim1"],
            claim2=case["claim2"],
            best_threshold=best_threshold
        )

        if result["predicted_label"] == case["expected"]:
            correct_predictions += 1

    # Final summary
    accuracy = (correct_predictions / len(test_cases)) * 100
    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Correct predictions: {correct_predictions}/{len(test_cases)}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"{'='*70}\n")
