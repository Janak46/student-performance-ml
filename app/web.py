from __future__ import annotations

from flask import Flask, render_template, request

from model_utils import (
    GENDER_OPTIONS,
    JOB_OPTIONS,
    RATING_OPTIONS,
    STUDYTIME_OPTIONS,
    TRAVELTIME_OPTIONS,
    YES_NO_OPTIONS,
    build_explanation_graphs,
    build_model_input,
    build_reason_points,
    load_dataset,
    load_model,
    load_student_record,
    predict_grade,
    scale_prediction_to_total,
    SAMPLE_INPUT_PATH,
)


app = Flask(__name__)

MODEL_PAYLOAD = load_model()
DATAFRAME = load_dataset()
SAMPLE_RECORD = load_student_record(SAMPLE_INPUT_PATH)

FORM_FIELDS = [
    {
        "name": "name",
        "label": "Student name",
        "type": "text",
        "help": "Used only in the result card so the prediction feels personal.",
    },
    {
        "name": "section",
        "label": "Section",
        "type": "text",
        "help": "Class or section label such as A, B, or 10-C.",
    },
    {
        "name": "subject",
        "label": "Subject",
        "type": "text",
        "help": "Subject for which you want the final score prediction.",
    },
    {
        "name": "total_marks",
        "label": "Total marks for the subject",
        "type": "number",
        "help": "Enter the full marks for this subject. It can be any positive number used by your institution.",
        "min": 1,
        "step": "any",
        "placeholder": "ex. 20, 50 or 100",
    },
    {
        "name": "G1",
        "label": "Test 1 marks",
        "type": "number",
        "help": "Marks from the first test, entered using the chosen total marks.",
        "min": 0,
    },
    {
        "name": "G2",
        "label": "Test 2 marks",
        "type": "number",
        "help": "Marks from the second test, entered using the chosen total marks.",
        "min": 0,
    },
    {
        "name": "absences",
        "label": "Absences",
        "type": "number",
        "help": "Total number of absences in this subject or term.",
        "min": 0,
        "max": 75,
    },
    {
        "name": "health",
        "label": "Health level",
        "type": "select",
        "help": "Self-reported health on a 1 to 5 scale.",
        "options": RATING_OPTIONS,
    },
    {
        "name": "traveltime",
        "label": "Travel time to school",
        "type": "select",
        "help": "Choose the average one-way travel time.",
        "options": TRAVELTIME_OPTIONS,
    },
    {
        "name": "studytime",
        "label": "Weekly study hours",
        "type": "select",
        "help": "Choose the closest range for weekly study time outside class.",
        "options": STUDYTIME_OPTIONS,
    },
    {
        "name": "sex",
        "label": "Gender",
        "type": "select",
        "help": "Included because the training dataset contains this field.",
        "options": GENDER_OPTIONS,
    },
    {
        "name": "Mjob",
        "label": "Mother's job",
        "type": "select",
        "help": "General job category from the training dataset.",
        "options": JOB_OPTIONS,
    },
    {
        "name": "Fjob",
        "label": "Father's job",
        "type": "select",
        "help": "General job category from the training dataset.",
        "options": JOB_OPTIONS,
    },
    {
        "name": "internet",
        "label": "Internet access at home",
        "type": "select",
        "help": "Whether the student has internet access at home.",
        "options": YES_NO_OPTIONS,
    },
    {
        "name": "freetime",
        "label": "Free time after school",
        "type": "select",
        "help": "Self-reported free time on a 1 to 5 scale.",
        "options": RATING_OPTIONS,
    },
]


def build_initial_values() -> dict:
    return {
        "name": SAMPLE_RECORD["name"],
        "section": SAMPLE_RECORD["section"],
        "subject": SAMPLE_RECORD["subject"],
        "total_marks": int(SAMPLE_RECORD["total_marks"]),
        "G1": float(SAMPLE_RECORD["G1"]),
        "G2": float(SAMPLE_RECORD["G2"]),
        "absences": int(SAMPLE_RECORD["absences"]),
        "health": int(SAMPLE_RECORD["health"]),
        "traveltime": int(SAMPLE_RECORD["traveltime"]),
        "studytime": int(SAMPLE_RECORD["studytime"]),
        "sex": SAMPLE_RECORD["sex"],
        "Mjob": SAMPLE_RECORD["Mjob"],
        "Fjob": SAMPLE_RECORD["Fjob"],
        "internet": SAMPLE_RECORD["internet"],
        "freetime": int(SAMPLE_RECORD["freetime"]),
    }


def parse_form_data(form_data) -> dict:
    values = {}
    for field in FORM_FIELDS:
        raw_value = form_data.get(field["name"], "")
        if field["name"] == "total_marks":
            values[field["name"]] = float(raw_value)
        elif field["type"] == "number":
            values[field["name"]] = float(raw_value)
        elif field["name"] in {"health", "traveltime", "studytime", "freetime"}:
            values[field["name"]] = int(raw_value)
        else:
            values[field["name"]] = raw_value
    return values


def validate(values: dict) -> list[str]:
    errors = []
    total_marks = values["total_marks"]

    for key in ["name", "section", "subject"]:
        if not str(values[key]).strip():
            errors.append(f"{key.title()} is required.")

    if total_marks <= 0:
        errors.append("Total marks must be greater than 0.")

    for test_key, label in [("G1", "Test 1 marks"), ("G2", "Test 2 marks")]:
        if values[test_key] < 0:
            errors.append(f"{label} cannot be negative.")
        if total_marks > 0 and values[test_key] > total_marks:
            errors.append(f"{label} cannot be greater than the chosen total marks.")

    if values["absences"] < 0:
        errors.append("Absences cannot be negative.")

    return errors


@app.route("/", methods=["GET", "POST"])
def index():
    values = build_initial_values()
    prediction = None
    normalized_prediction = None
    reason_points = []
    graphs = []
    errors = []

    if request.method == "POST":
        values = parse_form_data(request.form)
        errors = validate(values)

        if not errors:
            model_input = build_model_input(values, MODEL_PAYLOAD)
            normalized_prediction = predict_grade(model_input, MODEL_PAYLOAD)
            prediction = scale_prediction_to_total(
                normalized_prediction,
                values["total_marks"],
                MODEL_PAYLOAD,
            )
            reason_points = build_reason_points(DATAFRAME, model_input, values, MODEL_PAYLOAD)
            graphs = build_explanation_graphs(DATAFRAME, model_input, normalized_prediction)

    return render_template(
        "index.html",
        fields=FORM_FIELDS,
        values=values,
        prediction=prediction,
        normalized_prediction=normalized_prediction,
        graphs=graphs,
        reason_points=reason_points,
        errors=errors,
        model_name=MODEL_PAYLOAD["model_name"],
    )


if __name__ == "__main__":
    app.run(debug=True)
