README + requirements
pip freeze > requirements.txt (pin everything), then the README — the artifact interviewers read first:

House Price Prediction (PRCP-1020)
Predicting Ames, Iowa sale prices from 79 property features (1,458 sales, 2006–2010).

Approach
NA-safe loading + dtype audit → 3-group missingness classification → ordinalencoding + 9 engineered features → single sklearn Pipeline (impute/scale/one-hot,log-target) → 8-model comparison → tuned ElasticNet.

Results
CV log-RMSE 0.1146 | Holdout 0.1081 | MAE ≈ $14.1K | MAPE 8.0%

Key Insights

Simple models performed best: ElasticNet outperformed Random Forest (0.1319) and slightly beat XGBoost (0.1160) on cross-validation. Good feature engineering helped capture important patterns.

Context matters: ExterQual was important on its own but added little after accounting for overall quality. OverallCond showed a stronger positive relationship after accounting for house age.

Data cleaning was critical: Missing values were stored as "NA" strings. Distinguishing true missing data from genuine categories such as absence was essential for reliable analysis.

Limitations
One city; 2006–2010 sales window (crash-era); associations not causal; holdout n=292.

How to run
pip install -r requirements.txt, open notebooks/, Restart & Run All.

Adapt the insights into your own words — they must be speakable, not just hosted.