from __future__ import annotations

import argparse
from pathlib import Path

from model_utils import (
    MODEL_PATH,
    SAMPLE_INPUT_PATH,
    build_model_input,
    load_model,
    load_student_record,
    predict_grade,
    scale_prediction_to_total,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict a student's final subject score using the saved ML pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SAMPLE_INPUT_PATH,
        help="Path to a JSON file containing one student record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_model()
    form_values = load_student_record(args.input)
    model_input = build_model_input(form_values, payload)
    normalized_prediction = predict_grade(model_input, payload)
    scaled_prediction = scale_prediction_to_total(
        normalized_prediction,
        form_values["total_marks"],
        payload,
    )

    print(f"Loaded model artifact: {MODEL_PATH}")
    print(f"Saved model name: {payload['model_name']}")
    print(f"Loaded student input from: {args.input}")
    print(f"Student: {form_values['name']} ({form_values['section']})")
    print(f"Subject: {form_values['subject']}")
    print(
        f"Predicted final score: {scaled_prediction:.2f} / {form_values['total_marks']}"
    )


if __name__ == "__main__":
    main()
