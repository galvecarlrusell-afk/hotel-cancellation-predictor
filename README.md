# Hotel Booking Cancellation Prediction System

CpE Elective 1 Final Project | 1st Semester A.Y. 2026-2027 | Caraga State University, Cabadbaran Campus

**Authors:** Carl Rusell Q. Galve, Jetro C. Melloria, Riztel B. Monzolin
**Section:** CKUL3/CVHI3
**Live demo:** https://hotel-cancellation-predictor-duja7gm82vrnvhulaugxck.streamlit.app
**Repository:** https://github.com/galvecarlrusell-afk/hotel-cancellation-predictor

## Problem
Hotels lose revenue when guests cancel late, because rooms go unsold and staff plan around
bookings that never happen. This system estimates how likely a booking is to be cancelled so
staff can act early (reminders, deposits, overbooking plans).
Intended users: hotel managers, front-desk staff, booking coordinators.

## Dataset
- **Name:** Hotel Reservations Dataset (Kaggle), by Kaggle user ahsan81
- **Link:** https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
- **Original records:** 36,275 bookings, 19 columns (32.8% cancelled)
- **Records after cleaning:** 36,058 (train 28,846 | test 7,212)
- **Target:** `booking_status` (Canceled = 1, Not_Canceled = 0)
- **Features (15):** adults, children, weekend nights, week nights, meal plan, parking,
  room type, lead time, arrival month, market segment, repeated guest, previous cancellations,
  previous bookings not cancelled, average room price, special requests.
- **Data preparation:** checked for duplicate rows (none found); dropped `Booking_ID`,
  `arrival_year` and `arrival_date`; removed bookings with 0 adults (139 rows) or 0 nights
  (78 rows); scaled numeric features; one-hot encoded categorical features; stratified 80/20
  train-test split (random_state=42). No missing values were found.
- The CSV is not uploaded to this repository. Download it from the link above and save it
  as `data/hotel_reservations.csv`.

## Models
A baseline and four models, each inside one pipeline (scaling + encoding + classifier):
Logistic Regression, KNN (15 neighbours), Random Forest (150 trees) and Gradient Boosting.
Logistic Regression, Random Forest and Gradient Boosting use balanced class weights because
cancellations are the minority class. Settings are fixed (no tuning).

**Test-set results** (20% of the cleaned data, Canceled is the positive class):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (always Not Canceled) | 0.6717 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3283 |
| Logistic Regression | 0.7894 | 0.6472 | 0.7880 | 0.7107 | 0.8719 | 0.7745 |
| KNN | 0.8551 | 0.8047 | 0.7378 | 0.7698 | 0.9183 | 0.8517 |
| Random Forest | 0.8891 | 0.8200 | 0.8484 | 0.8340 | 0.9535 | 0.9250 |
| Gradient Boosting | 0.8799 | 0.7915 | 0.8611 | 0.8248 | 0.9507 | 0.9177 |

**5-fold stratified cross-validation on the training set** (mean F1 over the 5 folds), so the
ranking does not depend on one lucky split:

| Model | CV F1 (mean) |
|---|---|
| Logistic Regression | 0.690 |
| KNN | 0.748 |
| Random Forest | 0.826 |
| Gradient Boosting | 0.810 |

(Recall, precision, ROC-AUC and the standard deviations are in `results/cv_results.csv` and in the app's Model results tab.)

**Best model:** Random Forest, chosen by the highest cross-validated F1 (without using the test set)
and confirmed on the test set. Random Forest and Gradient Boosting are close: Gradient Boosting
catches slightly more cancellations (recall 0.861 vs 0.848) but has more false alarms (precision 0.792 vs 0.820).

**Why F1 and recall?** About 67% of bookings are not cancelled, so a baseline that never
predicts a cancellation already gets about 67% accuracy but catches no cancellations. Recall
(share of real cancellations caught) and F1 show whether a model is actually useful.

**Feature importance.** The Random Forest's built-in importance is shown in the app. The
Data insights tab shows the cancel rate by lead time, market segment, special requests and month.

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
python train.py                  # cross-validates, trains and saves models/ and results/
python -m streamlit run app.py   # opens the app at http://localhost:8501
python -m pytest                 # runs the tests
```
The saved files in `models/` were produced with scikit-learn 1.9.1, so keep that pin (or
re-run `train.py` after changing versions).

## Using the app
1. **Single booking:** enter the details (or load a low-risk / high-risk example) and click
   Predict. You get the prediction (Likely to Cancel / Likely to Stay), the cancellation
   probability, a LOW / MEDIUM / HIGH badge, a suggested action, the revenue at risk, a chart of
   which inputs push the risk up or down compared with a typical booking, the estimate from every
   model, and a what-if table.
2. **Batch (CSV):** download the template, fill in bookings, upload, filter by risk level, and
   download the predictions. Rows are colour-coded by risk.
3. **Data insights:** cancellation rate by lead time, market segment, special requests and month.
4. **Model results:** test and cross-validation tables, a plain-language explanation of each
   metric, confusion matrix, feature importance, and ROC / precision-recall curves.
5. **Sidebar:** switch models, choose the currency symbol, and adjust the MEDIUM and HIGH cut-offs.

Validation: the app rejects bookings with no adults, a stay under 1 night or a price of 0 or
less, and the batch upload reports missing columns and skips invalid rows without breaking the
other tabs.

## Files
- `train.py` - cleans the data, cross-validates, trains and saves the models and results
- `app.py` - Streamlit interface that loads the saved models
- `utils.py` - risk levels, validation and explanation helpers (no Streamlit, so they can be tested)
- `explore.py` - quick look at the raw data (missing values, duplicates, cancel rate)
- `tests/test_logic.py` - tests for the helpers, the saved models and the saved result files
- `models/` - saved models and `info.json` (column lists, dropdown options, typical values)
- `results/` - comparison and cross-validation tables, plots, feature importance, insights
- `.streamlit/config.toml` - app theme

## Limitations and ethics
- Predictions are estimates from historical data, not guarantees. The data comes from one hotel
  group, has no arrival year, and may not generalize to other hotels or to changed market conditions.
- The default risk levels (LOW below 35%, MEDIUM 35-59%, HIGH 60% or more) are fixed cut-offs we
  chose for the demo. They can be changed in the sidebar; a real hotel should set them from the
  cost of a missed cancellation versus a needless reminder.
- The "why this estimate" chart and the what-if table describe how the model's output changes.
  They do not prove that changing an input would prevent a cancellation.
- The model learns from past guest behaviour, including market segment. It should support staff
  decisions (a reminder, a deposit request), not be used to refuse or penalise guests
  automatically. Any action should be applied consistently and a person should stay in the loop.
- 545 bookings in the data have a room price of 0 (complimentary stays, 99% not cancelled).
  They are kept for training, but the app asks for a price above 0 so staff enter real bookings.
- The dataset contains no personal details (names, contacts), only booking attributes.

## AI tools used
We used Claude (Anthropic) to help draft and improve the code (data cleaning, model training,
the Streamlit app and the tests) and to troubleshoot setup and deployment. We chose the project
and dataset, ran and tested the training and the app, reviewed the files, set up GitHub and deployed the app to
Streamlit Cloud.