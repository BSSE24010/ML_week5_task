# Week 5: Improving a Classifier on a Rare Class (Loan Approval)

**Dataset:** `loan_prediction.csv` - 614 loan applications with 11 applicant/loan columns and the target `Loan_Status` (Y = approved, N = rejected). This is the well-known "Loan Prediction" practice dataset (Analytics Vidhya); the file is included in this repo.

## Summary
- The rare class is **rejected loans (N), 31% of the data**. A model that always says "approved" gets 69% accuracy but catches 0 rejected loans.
- A default RandomForest misses about half of the rejected loans. Error analysis shows that every missed case has a good credit history, so the other columns carry little signal about why those loans were rejected.
- I tried class weights, lower thresholds, oversampling (training folds only), new features (loan-to-income, EMI) and a leaf-size limit, one change at a time, chosen with cross-validation on the training data only.

## Result (test set, evaluated once, rejected class)

| Version | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy (most_frequent) | 0.691 | 0.000 | 0.000 | 0.000 |
| Baseline RandomForest | 0.780 | 0.690 | 0.526 | 0.597 |
| Final (new features + min_samples_leaf=5, threshold 0.3) | 0.683 | 0.488 | 0.553 | 0.519 |

The final model caught one more rejected loan (21 vs 20 of 38) but raised false alarms from 9 to 22. The improvement seen in cross-validation did **not** carry over to the test set, so the baseline is the better choice here. Details and reasons are in the notebook.

## How to run
```bash
pip install -r requirements.txt
jupyter notebook loan_model_improvement.ipynb
```
Then use **Kernel > Restart and Run All**. Keep `loan_prediction.csv` in the same folder. All randomness uses `random_state=42`.
