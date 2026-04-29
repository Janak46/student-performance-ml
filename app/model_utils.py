from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


matplotlib.use("Agg")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "best_student_grade_model.joblib"
DATA_PATH = BASE_DIR / "data" / "student_data.csv"
SAMPLE_INPUT_PATH = Path(__file__).resolve().with_name("sample_student.json")
DEFAULT_MODEL_SCALE = 20

STUDYTIME_OPTIONS = [
    {"value": 1, "label": "Less than 2 hours per week"},
    {"value": 2, "label": "2 to 5 hours per week"},
    {"value": 3, "label": "5 to 10 hours per week"},
    {"value": 4, "label": "More than 10 hours per week"},
]

TRAVELTIME_OPTIONS = [
    {"value": 1, "label": "Less than 15 minutes"},
    {"value": 2, "label": "15 to 30 minutes"},
    {"value": 3, "label": "30 minutes to 1 hour"},
    {"value": 4, "label": "More than 1 hour"},
]

RATING_OPTIONS = [
    {"value": 1, "label": "1 - Very low"},
    {"value": 2, "label": "2 - Low"},
    {"value": 3, "label": "3 - Medium"},
    {"value": 4, "label": "4 - Good"},
    {"value": 5, "label": "5 - Very high"},
]

JOB_OPTIONS = [
    {"value": "teacher", "label": "Teacher"},
    {"value": "services", "label": "Services"},
    {"value": "health", "label": "Healthcare"},
    {"value": "at_home", "label": "Works at home"},
    {"value": "other", "label": "Other"},
]

YES_NO_OPTIONS = [
    {"value": "yes", "label": "Yes"},
    {"value": "no", "label": "No"},
]

GENDER_OPTIONS = [
    {"value": "F", "label": "Female"},
    {"value": "M", "label": "Male"},
]


def load_model() -> dict:
    return joblib.load(MODEL_PATH)


def load_student_record(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def model_scale(payload: dict) -> int:
    return int(payload.get("target_scale_max", DEFAULT_MODEL_SCALE))


def normalize_marks(raw_marks: float, total_marks: float, payload: dict) -> float:
    scale_max = model_scale(payload)
    return (raw_marks / total_marks) * scale_max


def scale_prediction_to_total(prediction: float, total_marks: float, payload: dict) -> float:
    scale_max = model_scale(payload)
    scaled = (prediction / scale_max) * total_marks
    return max(0.0, min(float(total_marks), scaled))


def build_model_input(form_values: dict, payload: dict) -> dict:
    total_marks = float(form_values["total_marks"])
    return {
        "sex": form_values["sex"],
        "Mjob": form_values["Mjob"],
        "Fjob": form_values["Fjob"],
        "traveltime": int(form_values["traveltime"]),
        "studytime": int(form_values["studytime"]),
        "internet": form_values["internet"],
        "freetime": int(form_values["freetime"]),
        "health": int(form_values["health"]),
        "absences": int(form_values["absences"]),
        "G1": normalize_marks(float(form_values["G1"]), total_marks, payload),
        "G2": normalize_marks(float(form_values["G2"]), total_marks, payload),
    }


def predict_grade(student_record: dict, payload: dict) -> float:
    pipeline = payload["pipeline"]
    input_df = pd.DataFrame([student_record])
    prediction = pipeline.predict(input_df)[0]
    return float(prediction)


def _figure_to_base64() -> str:
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, user_x: float, user_y: float, title: str, xlabel: str) -> dict:
    plt.figure(figsize=(4.8, 3.4))
    plt.scatter(df[x_col], df[y_col], alpha=0.45, color="#1e7a5d")
    plt.scatter([user_x], [user_y], color="#d84f2a", s=90, label="Student")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Final score (out of 20)")
    plt.legend()
    return {"title": title, "image": _figure_to_base64()}


def _average_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, user_value: int, title: str, xlabel: str) -> dict:
    summary = df.groupby(x_col)[y_col].mean().reset_index()
    plt.figure(figsize=(4.8, 3.4))
    plt.bar(summary[x_col].astype(str), summary[y_col], color="#f2b544")
    if user_value in summary[x_col].tolist():
        index = summary[x_col].tolist().index(user_value)
        plt.bar(str(summary.iloc[index][x_col]), summary.iloc[index][y_col], color="#1e7a5d")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Average final score (out of 20)")
    return {"title": title, "image": _figure_to_base64()}


def build_explanation_graphs(df: pd.DataFrame, student_input: dict, predicted_score_20: float) -> list[dict]:
    return [
        _scatter_plot(
            df,
            "G1",
            "G3",
            student_input["G1"],
            predicted_score_20,
            "Test 1 marks vs final score",
            "Test 1 marks (out of 20)",
        ),
        _scatter_plot(
            df,
            "G2",
            "G3",
            student_input["G2"],
            predicted_score_20,
            "Test 2 marks vs final score",
            "Test 2 marks (out of 20)",
        ),
        _average_bar_plot(
            df,
            "studytime",
            "G3",
            int(student_input["studytime"]),
            "Study time and average final score",
            "Study time category",
        ),
        _scatter_plot(
            df,
            "absences",
            "G3",
            student_input["absences"],
            predicted_score_20,
            "Absences vs final score",
            "Number of absences",
        ),
    ]


def build_reason_points(df: pd.DataFrame, student_input: dict, form_values: dict, payload: dict) -> list[str]:
    average_g1 = df["G1"].mean()
    average_g2 = df["G2"].mean()
    average_absences = df["absences"].mean()
    average_studytime = df["studytime"].mean()
    internet_text = "has" if student_input["internet"] == "yes" else "does not have"
    scale_max = model_scale(payload)

    return [
        (
            f"The model relies most on the two earlier test scores. This student's normalized marks "
            f"are {student_input['G1']:.1f}/{scale_max} and {student_input['G2']:.1f}/{scale_max}, "
            f"compared with class averages of {average_g1:.1f}/{scale_max} and {average_g2:.1f}/{scale_max}."
        ),
        (
            f"Absences matter because students with fewer missed classes usually score better. "
            f"This student has {student_input['absences']} absences versus an average of {average_absences:.1f}."
        ),
        (
            f"Study time adds context to the marks. The chosen study-time level is {int(student_input['studytime'])} "
            f"while the dataset average is {average_studytime:.1f}, and the student {internet_text} home internet access."
        ),
        (
            f"The predicted result is scaled to the subject total chosen by the user. "
            f"For {form_values['subject']}, the app predicts the final score out of {form_values['total_marks']}."
        ),
    ]
