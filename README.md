# ML_CW1-Data-Challenge

Coursework project for ML modeling. Includes notebooks for exploration/evaluation and a standalone script for the MLP model pipeline.

## Contents
- best_model.py: End-to-end data prep + MLP training + submission export
- CW1_model_exploration.ipynb / CW1_model_eval.ipynb: Notebooks for experimentation
- CW1_train.csv / CW1_test.csv: Input data
- CW1_submission_K23067889.csv: Example submission output

## Setup
Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the MLP pipeline
This trains the MLP, prints train/val R2 and generalization gap, then writes the submission CSV.

```powershell
python best_model.py
```

Output:
- CW1_submission_K23067889.csv

## Notes
- The script expects the CSVs to be in the project root.
- You can change the output filename in best_model.py.
