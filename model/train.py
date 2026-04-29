from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "student_data.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
TARGET_COLUMN = "G3"
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "sex",
    "Mjob",
    "Fjob",
    "traveltime",
    "studytime",
    "internet",
    "freetime",
    "health",
    "absences",
    "G1",
    "G2",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset from: {DATA_PATH}")
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def evaluate_model(model_name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    pipeline.fit(X_train, y_train)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    results = {
        "name": model_name,
        "pipeline": pipeline,
        "train_r2": r2_score(y_train, train_predictions),
        "test_r2": r2_score(y_test, test_predictions),
        "test_mae": mean_absolute_error(y_test, test_predictions),
        "test_rmse": mean_squared_error(y_test, test_predictions) ** 0.5,
        "predictions": test_predictions,
    }

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Train R^2: {results['train_r2']:.3f}")
    print(f"Test R^2:  {results['test_r2']:.3f}")
    print(f"Test MAE:  {results['test_mae']:.3f}")
    print(f"Test RMSE: {results['test_rmse']:.3f}")

    comparison = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(test_predictions).round(2),
        }
    )
    comparison["error"] = (comparison["actual"] - comparison["predicted"]).round(2)
    print("\nSample predictions:")
    print(comparison.head(5).to_string(index=False))

    return results


def print_linear_regression_insights(pipeline: Pipeline) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    coefficients = (
        pd.DataFrame({"feature": feature_names, "coefficient": model.coef_})
        .assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
    )

    print("\nTop Linear Regression coefficients:")
    print(coefficients[["feature", "coefficient"]].head(10).to_string(index=False))


def print_random_forest_insights(pipeline: Pipeline) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    importance = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )

    print("\nTop Random Forest feature importances:")
    print(importance.head(10).to_string(index=False))


def tune_random_forest(preprocessor: ColumnTransformer, X_train, y_train) -> GridSearchCV:
    tuning_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=tuning_pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=5,
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    print("\nRandom Forest tuning")
    print("--------------------")
    print(f"Best cross-validation R^2: {search.best_score_:.3f}")
    print(f"Best parameters: {search.best_params_}")

    return search


def save_model(pipeline: Pipeline, model_name: str) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "best_student_grade_model.joblib"
    payload = {
        "model_name": model_name,
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "target_scale_max": 20,
        "pipeline": pipeline,
    }
    joblib.dump(payload, model_path)
    print(f"\nSaved best model to: {model_path}")
    return model_path


def main() -> None:
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    print(f"Target column: {TARGET_COLUMN}")
    print(f"Selected features: {FEATURE_COLUMNS}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Train split: {X_train.shape[0]} rows")
    print(f"Test split:  {X_test.shape[0]} rows")

    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    linear_results = evaluate_model(
        "Linear Regression Baseline",
        linear_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    print_linear_regression_insights(linear_results["pipeline"])

    forest_results = evaluate_model(
        "Random Forest Comparison",
        forest_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    print_random_forest_insights(forest_results["pipeline"])

    tuning_search = tune_random_forest(preprocessor, X_train, y_train)
    tuned_forest_results = evaluate_model(
        "Tuned Random Forest",
        tuning_search.best_estimator_,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    print_random_forest_insights(tuned_forest_results["pipeline"])

    results_df = pd.DataFrame(
        [
            {
                "model": linear_results["name"],
                "test_r2": linear_results["test_r2"],
                "test_mae": linear_results["test_mae"],
                "test_rmse": linear_results["test_rmse"],
            },
            {
                "model": forest_results["name"],
                "test_r2": forest_results["test_r2"],
                "test_mae": forest_results["test_mae"],
                "test_rmse": forest_results["test_rmse"],
            },
            {
                "model": tuned_forest_results["name"],
                "test_r2": tuned_forest_results["test_r2"],
                "test_mae": tuned_forest_results["test_mae"],
                "test_rmse": tuned_forest_results["test_rmse"],
            },
        ]
    ).sort_values("test_r2", ascending=False)

    best_model = results_df.iloc[0]["model"]
    best_pipeline = {
        linear_results["name"]: linear_results["pipeline"],
        forest_results["name"]: forest_results["pipeline"],
        tuned_forest_results["name"]: tuned_forest_results["pipeline"],
    }[best_model]
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model on test R^2: {best_model}")
    save_model(best_pipeline, best_model)


if __name__ == "__main__":
    main()
