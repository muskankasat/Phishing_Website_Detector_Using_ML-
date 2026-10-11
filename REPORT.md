# Explainable Phishing Website Risk Detector — Project Report

## 1. Problem (Cybersecurity)
Phishing websites imitate legitimate sites to steal credentials or financial data. This project builds and compares eight machine-learning classifiers, from classic baselines to modern gradient-boosting and neural models, that label a website record as *Phishing* or *Legitimate*. A Streamlit app presents predictions, performance, cross-validation and literature comparison.

**Scope note:** the models work on 30 pre-extracted, encoded features. This is not a live URL scanner.

## 2. Dataset
- Source: UCI Machine Learning Repository, *Phishing Websites* (id 327), CC BY 4.0. Collected from PhishTank, MillerSmiles and Google search operators.
- 11,055 rows, 30 integer features (values -1, 0, 1), binary target (-1 = phishing, 1 = legitimate). Raw classes: 4,898 phishing, 6,157 legitimate.
- Missing values: 0.
- Exact duplicate feature rows: 5,270. About 71% of raw rows have an exact twin elsewhere in the data (see `experiments/duplicate_effect.py`).
- After removing duplicates: **5,849 rows** (3,019 phishing, 2,830 legitimate), nearly balanced.

## 3. Method
1. Map labels, coerce features to numeric, median-impute gaps (none needed), drop duplicates.
2. Stratified 80:20 split, `random_state=42`: 4,679 train and 1,170 test rows. The split happens before any fitting.
3. Models:
   - **Classic:** Logistic Regression (scaled), Decision Tree, Random Forest (200 trees), Gradient Boosting.
   - **Modern:** XGBoost, LightGBM, CatBoost, and an MLP neural network (64-32 hidden units, scaled inputs, early stopping).
4. Evaluation: hold-out metrics (accuracy, precision, recall, F1 for phishing, macro F1, ROC-AUC), confusion matrices, and 5-fold stratified cross-validation on the training split only (the test set is never used for CV).
5. No hyperparameter tuning was done; all models use sensible fixed settings in `phishing_models.py`.

## 4. Results (hold-out test set, 1,170 rows)
| Model | Type | Accuracy | Precision (Phish) | Recall (Phish) | F1 (Phish) | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| CatBoost | Advanced | 0.9513 | 0.9675 | 0.9371 | 0.9521 | 0.9513 | 0.9928 |
| LightGBM | Advanced | 0.9504 | 0.9659 | 0.9371 | 0.9513 | 0.9504 | 0.9932 |
| XGBoost | Advanced | 0.9487 | 0.9564 | 0.9437 | 0.9500 | 0.9487 | 0.9937 |
| Gradient Boosting | Classic | 0.9444 | 0.9560 | 0.9354 | 0.9456 | 0.9444 | 0.9892 |
| Random Forest | Classic | 0.9402 | 0.9572 | 0.9255 | 0.9411 | 0.9402 | 0.9856 |
| MLP (Neural Network) | Advanced | 0.9325 | 0.9382 | 0.9305 | 0.9343 | 0.9324 | 0.9861 |
| Decision Tree | Classic | 0.9162 | 0.9303 | 0.9056 | 0.9178 | 0.9162 | 0.9237 |
| Logistic Regression | Classic | 0.9154 | 0.9244 | 0.9106 | 0.9174 | 0.9153 | 0.9777 |

Training time is saved in `results/metrics.csv` and varies by machine.

### Confusion matrices (604 phishing / 566 legitimate)
| Model | Phishing caught | Phishing missed (FN) | Legitimate flagged (FP) | Legitimate correct |
|---|---|---|---|---|
| CatBoost | 566 | 38 | 19 | 547 |
| LightGBM | 566 | 38 | 20 | 546 |
| XGBoost | 570 | 34 | 26 | 540 |
| Gradient Boosting | 565 | 39 | 26 | 540 |
| Random Forest | 559 | 45 | 25 | 541 |
| MLP (Neural Network) | 562 | 42 | 37 | 529 |
| Decision Tree | 547 | 57 | 41 | 525 |
| Logistic Regression | 550 | 54 | 45 | 521 |

### 5-fold cross-validation (training split)
| Model | Accuracy (mean ± std) | Phishing recall (mean ± std) | Macro F1 (mean ± std) |
|---|---|---|---|
| CatBoost | 0.9500 ± 0.0036 | 0.9474 ± 0.0085 | 0.9499 ± 0.0036 |
| XGBoost | 0.9498 ± 0.0070 | 0.9499 ± 0.0132 | 0.9497 ± 0.0070 |
| Random Forest | 0.9474 ± 0.0047 | 0.9433 ± 0.0084 | 0.9474 ± 0.0047 |
| LightGBM | 0.9470 ± 0.0045 | 0.9458 ± 0.0088 | 0.9469 ± 0.0045 |
| Gradient Boosting | 0.9444 ± 0.0053 | 0.9400 ± 0.0116 | 0.9444 ± 0.0053 |
| MLP (Neural Network) | 0.9301 ± 0.0033 | 0.9321 ± 0.0133 | 0.9300 ± 0.0033 |
| Decision Tree | 0.9246 ± 0.0057 | 0.9197 ± 0.0103 | 0.9245 ± 0.0057 |
| Logistic Regression | 0.9207 ± 0.0026 | 0.9139 ± 0.0107 | 0.9207 ± 0.0026 |

The top four models (CatBoost, XGBoost, Random Forest, LightGBM) are within about 0.3 points of each other, which is smaller than their fold-to-fold standard deviation (0.4-0.7 points). Their ranking is therefore not statistically reliable.

### Feature importance (Random Forest)
`SSLfinal_State` (0.321), `URL_of_Anchor` (0.238), `having_Sub_Domain` (0.059), `Prefix_Suffix` (0.054), `web_traffic` (0.052). The top two features together account for 55.8% of total importance. This describes what the fitted forest relied on, not causes of phishing.

## 5. Comparison with published work
| Source | Model(s) | Reported result | Dataset note |
|---|---|---|---|
| [Phishing Attacks and Websites Classification Using Machine Learning and Multiple Datasets (arXiv 2101.02552)](https://arxiv.org/pdf/2101.02552) | Random Forest, Artificial Neural Network | over 97% accuracy | UCI Phishing Websites, 11,055 instances, 30 features |
| [A Sophisticated Framework for the Accurate Detection of Phishing Websites (arXiv 2403.09735)](https://arxiv.org/pdf/2403.09735) | Proposed algorithm | 97.49% accuracy | Dataset-1 = UCI Phishing Websites (11,055 rows) |
| [Comparative evaluation of machine learning algorithms for phishing site detection (PMC11232597)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11232597/) | XGBoost (after hyperparameter tuning); Random Forest | XGBoost 98% accuracy; Random Forest 96% accuracy | UCI dataset (11,055 records) |
| [Phishing website prediction using base and ensemble classifier techniques with cross-validation (Cybersecurity, Springer)](https://link.springer.com/article/10.1186/s42400-022-00126-9) | Extra Trees, XGBoost | best result 99.18% accuracy across two datasets | One dataset has 2,456 instances, the other 11,055; mapping of each score to dataset not verified |

Published studies report roughly 96-99% accuracy on this dataset, higher than the 91-95% reported here. The difference is largely explained by duplicates. Re-running our eight models with duplicates kept (same split settings, new 80:20 split of all 11,055 rows):

| Model | Accuracy, duplicates kept (2,211 test rows) | Accuracy, de-duplicated (1,170 test rows) | Drop (points) |
|---|---|---|---|
| LightGBM | 0.9769 | 0.9504 | 2.65 |
| Random Forest | 0.9765 | 0.9402 | 3.63 |
| XGBoost | 0.9751 | 0.9487 | 2.64 |
| CatBoost | 0.9733 | 0.9513 | 2.20 |
| Decision Tree | 0.9692 | 0.9162 | 5.30 |
| MLP (Neural Network) | 0.9656 | 0.9325 | 3.31 |
| Gradient Boosting | 0.9502 | 0.9444 | 0.58 |
| Logistic Regression | 0.9317 | 0.9154 | 1.63 |

With duplicates kept, LightGBM, Random Forest, XGBoost and CatBoost reach about 97.3-97.7%, in line with the published range; after de-duplication they drop by about 2-4 points. Identical records appearing in both training and test sets inflate accuracy, so the de-duplicated numbers are the more honest estimate of generalization. Caveat: the two columns use different test sets, and the cited papers use different splits and preprocessing, so this is an explanation consistent with the evidence, not a controlled replication of any paper.

## 6. Discussion
- CatBoost, LightGBM and XGBoost lead on the hold-out set (CatBoost best at 0.9513 accuracy), ahead of the classic ensembles (Gradient Boosting, Random Forest) by about 0.4-1.1 points, and clearly ahead of the Decision Tree and Logistic Regression (about 3.5 points).
- The MLP trails the tree ensembles (0.9325), consistent with tree models usually being strong on tabular, categorical-style features.
- Logistic Regression has a high ROC-AUC (0.978) for its accuracy: it ranks well but its 0.5 threshold is less suited to the data.
- Every model still misses roughly 6-9% of phishing records. The highest phishing recall on the test set is 0.9437 (XGBoost). For a security tool, false negatives are the costlier error, so recall deserves as much attention as accuracy.

## 7. Limitations
- Features are engineered and encoded; no raw URLs or page content are used.
- The dataset dates from 2012-2015; current phishing techniques may differ.
- One hold-out split plus CV on the training split; no hyperparameter tuning; no external validation set.
- Literature numbers are reported by other authors under different conditions; they are indicative only.
- Model scores are not calibrated probabilities. The app does not fetch or verify live websites.

## 8. Future work
Hyperparameter tuning with nested CV; threshold selection to raise phishing recall; probability calibration; evaluation on a newer independent dataset; a secure, validated live feature-extraction pipeline.

## 9. Reproducibility
```
pip install -r requirements.txt
python train.py
python -m experiments.duplicate_effect
ruff check . && pytest
streamlit run app.py
```
Fixed `random_state=42`. All numbers above come from `results/*.csv` generated by the scripts.

## 10. References
- Mohammad, R., Thabtah, F., McCluskey, L. *Phishing Websites*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/327/phishing+websites
- Sources listed in Section 5 (also in `results/literature_comparison.csv`).
