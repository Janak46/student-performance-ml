# student-performance-ml

This is a beginner machine learning project that predicts student final grades (`G3`).
It now includes preprocessing, model comparison, beginner-friendly random forest tuning, and saving the best model.

## What is inside

- `data/student_data.csv` contains the dataset
- `notebook/analysis.ipynb` walks through the learning workflow step by step
- `model/train.py` runs the full preprocessing, training, evaluation, tuning, and model comparison pipeline
- `artifacts/` stores the best saved model after training
- `app/predict.py` loads the saved model and predicts `G3` for one student record
- `app/web.py` starts a browser-based interface for entering student data and predicting `G3`

The web app now uses a smaller and clearer form. It focuses on:
- subject details
- total marks for the subject
- two earlier test scores
- absences, study time, travel time, free time, health, internet access
- a small number of background fields still required by the trained dataset

## Run the project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the training script from the project root:

```bash
python model/train.py
```

After training finishes, the best pipeline is saved to `artifacts/best_student_grade_model.joblib`.

## Make a prediction

Run the prediction script with the sample student input:

```bash
python app/predict.py
```

Or point it at your own JSON file:

```bash
python app/predict.py --input app/sample_student.json
```

## Run the web app

Start the Flask app:

```bash
python app/web.py
```

Then open the local address shown in the terminal, usually `http://127.0.0.1:5000`.
