# Viva preparation: Phishing Website Risk Detector

## Core questions
1. **What is the real-world problem?** Phishing sites imitate legitimate sites to trick users into sharing credentials or financial details.
2. **What is the target variable?** The dataset label, mapped in the code to `Phishing` or `Legitimate`.
3. **What is a feature?** An input attribute used by a model to make a prediction.
4. **Why did you choose this dataset?** It is a public, manageable, labeled classification dataset with engineered website features.
5. **Does this app scan any URL pasted by a user?** No. It classifies the dataset's encoded features. Live URL analysis would require a separate, validated feature-extraction pipeline.
6. **Why split the dataset?** Training data fits the model; held-out test data estimates performance on unseen examples.
7. **Why stratify?** To preserve approximate class proportions in both splits.
8. **Why use a fixed random state?** To make the split and results reproducible.
9. **Why compare eight models?** They cover linear, single-tree, bagged-tree, classic boosting, modern gradient boosting (XGBoost, LightGBM, CatBoost) and a neural network (MLP), so the comparison includes current standard tabular methods.
10. **What is accuracy?** Correct predictions divided by all predictions.
11. **What is precision?** Among items predicted phishing, the fraction that truly are phishing.
12. **What is recall?** Among truly phishing items, the fraction correctly detected.
13. **What is F1-score?** The harmonic mean of precision and recall.
14. **Why is phishing recall important?** Low recall means more phishing examples are missed and classified as legitimate.
15. **What is a false positive?** A legitimate site classified as phishing.
16. **What is a false negative?** A phishing site classified as legitimate.
17. **What does a confusion matrix show?** Counts of correct and incorrect predictions for each class.
18. **What is ROC-AUC?** A threshold-independent measure of how well scores rank positive examples above negative examples; it is computed here for the phishing class.
19. **Why scale Logistic Regression inputs?** Feature scaling often helps optimization and makes regularized linear models behave more consistently.
20. **Why not scale Decision Tree or Random Forest inputs?** Tree splits generally do not require features to be on comparable scales.
21. **What is class imbalance?** When one class has more examples than another. Accuracy can be misleading in that case.
22. **What does feature importance mean?** It describes which features the fitted Random Forest used more for its splits; it does not prove causation.
23. **How do you prevent data leakage?** Split first and fit models only on the training split; evaluate on the untouched test split.
24. **What are the limitations?** Dataset age, encoded features, possible dataset-specific patterns, and no validation on a fresh external dataset.
25. **What would you improve with more time?** Test on a newer independent dataset, perform cross-validation and tuning, calibrate probabilities, and design a secure live URL feature-extraction pipeline.

## Added questions
26. **Why did you remove duplicates?** About 71% of the raw 11,055 rows have an exact twin. Left in, identical records land in both training and test sets and inflate accuracy. After removal 5,849 rows remain.
27. **Papers report 97-98% but you got about 95%. Why?** With duplicates kept, our top models also reach about 97.3-97.7% (`results/duplicate_effect.csv`). After de-duplication they drop 2-4 points. Papers also differ in splits and preprocessing, so this is an explanation, not a proven replication.
28. **Why cross-validation, and on which data?** One split can mislead. 5-fold stratified CV on the training split only gives mean and standard deviation; the test set is never touched.
29. **Is CatBoost really better than XGBoost or Random Forest?** Not reliably. The top four models differ by about 0.3 points in CV, less than the fold-to-fold standard deviation.
30. **Why does the MLP trail tree models?** Features are small ternary codes and trees handle such tabular data well; the MLP was not tuned. Do not claim neural networks are inherently worse.
31. **Why wrap XGBoost and the MLP in a label-encoding wrapper?** They need integer labels, while the project keeps readable string labels for the UI. The wrapper converts in `fit` and back in `predict`.
32. **What do you mean by "state-of-the-art" here?** XGBoost, LightGBM and CatBoost are widely used strong tabular baselines and published phishing papers use them. We did not tune them or implement every recent paper, so we call them modern strong baselines.
33. **How do you test code quality?** `ruff` linting, pytest smoke tests (models, predictions, no duplicates, app runs) and GitHub Actions CI.

## Before the viva
- Know the exact row/feature counts from your run.
- Know which model scored best on macro F1 (CatBoost) and on phishing recall (XGBoost), and that the top models are statistically close.
- Be able to point to the false negatives in a confusion matrix.
- Be ready to explain the difference between a model score and a guarantee of safety.
- Never claim a metric that is not in your generated results.
