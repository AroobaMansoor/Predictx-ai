# PredictX AI — Experimental Results

_Generated automatically by `train.py` on 2026-11-06T17:54:49 (runtime 13.5 min, seed 42). All numbers below come from this run._

## Data split

| split | rows | failures | failure_rate |
|---|---|---|---|
| train | 6000.0000 | 203.0000 | 0.0338 |
| validation | 2000.0000 | 68.0000 | 0.0340 |
| test | 2000.0000 | 68.0000 | 0.0340 |

## EDA observations

- The dataset has 10,000 rows and 14 columns with 0 missing values and 0 fully duplicated rows.
- Failures are rare: 339 of 10,000 records (3.39%), an imbalance of about 1:28. Accuracy alone is therefore misleading (always predicting 'No Failure' would already score 96.6%).
- Failure rate by product type: L = 3.92% (n=6,000), M = 2.77% (n=2,997), H = 2.09% (n=1,003).
- Failed machines have higher mean torque (50.2 vs 39.6 Nm) and higher mean tool wear (144 vs 107 min).
- Mean temperature difference is lower for failures (9.40 vs 10.02 K), consistent with heat-dissipation problems.
- Rotational speed and torque are strongly negatively correlated (r = -0.88); air and process temperature are strongly positively correlated (r = 0.88).
- Strongest linear correlations with failure: Torque [Nm] (+0.191), Wear x Torque [min·Nm] (+0.190), Power [W] (+0.176). Linear correlations are weak overall, which suggests non-linear (threshold-like) failure behaviour.
- IQR outliers: Rotational speed [rpm] 418 rows (failure rate 8.4%), Torque [Nm] 69 rows (failure rate 89.9%) versus 3.39% overall. These are plausible extreme operating points that carry real failure signal, so they are kept rather than removed.
- Failure-mode labels overlap: 24 rows carry more than one mode, 9 failures have no mode label, and 18 rows have a mode flag but Machine failure = 0 (RNF=18).

## Experiment 2 — baseline vs engineered features (default hyperparameters)

| model | feature_set | cv_pr_auc_mean | cv_pr_auc_std | val_pr_auc | val_f1_at_val_threshold |
|---|---|---|---|---|---|
| Logistic Regression | baseline | 0.4721 | 0.0903 | 0.4054 | 0.4380 |
| Logistic Regression | engineered | 0.4997 | 0.0860 | 0.4682 | 0.4748 |
| Random Forest | baseline | 0.7671 | 0.0641 | 0.7688 | 0.7438 |
| Random Forest | engineered | 0.9014 | 0.0463 | 0.8649 | 0.8475 |
| XGBoost | baseline | 0.7813 | 0.0495 | 0.7103 | 0.6774 |
| XGBoost | engineered | 0.8818 | 0.0402 | 0.8300 | 0.7805 |
| LightGBM | baseline | 0.7891 | 0.0880 | 0.7895 | 0.7321 |
| LightGBM | engineered | 0.8861 | 0.0512 | 0.8589 | 0.8264 |
| CatBoost | baseline | 0.8194 | 0.0449 | 0.7617 | 0.7344 |
| CatBoost | engineered | 0.8937 | 0.0468 | 0.8592 | 0.8154 |
| Neural Network (MLP) | baseline | 0.7356 | 0.0734 | 0.7290 | 0.7077 |
| Neural Network (MLP) | engineered | 0.7925 | 0.0724 | 0.7722 | 0.7273 |

Chosen feature set: **engineered**

## Experiment 1 — tuned model comparison (validation set, used for selection)

| model | cv_pr_auc | val_pr_auc | val_roc_auc | val_precision | val_recall | val_f1 | val_threshold |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9203 | 0.8897 | 0.9834 | 1.0000 | 0.7353 | 0.8475 | 0.6960 |
| CatBoost | 0.8985 | 0.8622 | 0.9831 | 0.8966 | 0.7647 | 0.8254 | 0.2732 |
| LightGBM | 0.8900 | 0.8536 | 0.9851 | 0.9091 | 0.7353 | 0.8130 | 0.4208 |
| XGBoost | 0.8931 | 0.8489 | 0.9793 | 0.9123 | 0.7647 | 0.8320 | 0.7012 |
| Neural Network (MLP) | 0.7666 | 0.7119 | 0.9600 | 0.7885 | 0.6029 | 0.6833 | 0.4465 |
| Logistic Regression | 0.5012 | 0.4678 | 0.8981 | 0.4595 | 0.5000 | 0.4789 | 0.2058 |

Selected model: **Random Forest** (highest validation pr_auc).

## Experiment 4 — impact of hyperparameter tuning (validation)

| model | default_val_pr_auc | val_pr_auc | tuning_gain_pr_auc | default_val_f1 | val_f1 | tuning_gain_f1 |
|---|---|---|---|---|---|---|
| Random Forest | 0.8649 | 0.8897 | 0.0248 | 0.8475 | 0.8475 | 0.0000 |
| CatBoost | 0.8592 | 0.8622 | 0.0030 | 0.8154 | 0.8254 | 0.0100 |
| LightGBM | 0.8589 | 0.8536 | -0.0054 | 0.8264 | 0.8130 | -0.0134 |
| XGBoost | 0.8300 | 0.8489 | 0.0190 | 0.7805 | 0.8320 | 0.0515 |
| Neural Network (MLP) | 0.7722 | 0.7119 | -0.0603 | 0.7273 | 0.6833 | -0.0439 |
| Logistic Regression | 0.4682 | 0.4678 | -0.0004 | 0.4748 | 0.4789 | 0.0041 |

## Experiment 3 — class-imbalance strategies (Random Forest)

| strategy | cv_pr_auc_mean | val_pr_auc | val_f1_tuned_thr | val_recall_at_0.5 | val_precision_at_0.5 |
|---|---|---|---|---|---|
| none | 0.9203 | 0.8897 | 0.8475 | 0.7353 | 1.0000 |
| class_weight | 0.8949 | 0.8414 | 0.7899 | 0.9118 | 0.4397 |
| smote | 0.8897 | 0.8329 | 0.7899 | 0.8382 | 0.5135 |
| random_undersample | 0.6979 | 0.5806 | 0.5594 | 0.9265 | 0.2052 |

Decision: **as tuned (search included class weighting)**

## Experiment 5 — probability calibration (validation)

| calibration | val_brier | val_ece | val_log_loss | val_pr_auc | val_roc_auc | val_f1 | val_threshold |
|---|---|---|---|---|---|---|---|
| none | 0.0083 | 0.0044 | 0.0348 | 0.8897 | 0.9834 | 0.8475 | 0.6960 |
| sigmoid | 0.0088 | 0.0040 | 0.0454 | 0.8874 | 0.9812 | 0.8475 | 0.7513 |
| isotonic | 0.0085 | 0.0053 | 0.0346 | 0.8828 | 0.9833 | 0.8480 | 0.2701 |

Chosen: **none** (lowest Brier)

## Final held-out TEST results (thresholds tuned on validation)

| model | accuracy | precision | recall | f1 | roc_auc | pr_auc | false_positive_rate | false_negative_rate | threshold | single_row_latency_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9645 | 0.4795 | 0.5147 | 0.4965 | 0.9201 | 0.4614 | 0.0197 | 0.4853 | 0.2058 | 10.2326 |
| Random Forest | 0.9915 | 1.0000 | 0.7500 | 0.8571 | 0.9715 | 0.9032 | 0.0000 | 0.2500 | 0.6960 | 28.1357 |
| XGBoost | 0.9900 | 0.9000 | 0.7941 | 0.8438 | 0.9733 | 0.8867 | 0.0031 | 0.2059 | 0.7012 | 7.7259 |
| LightGBM | 0.9925 | 0.9206 | 0.8529 | 0.8855 | 0.9726 | 0.8981 | 0.0026 | 0.1471 | 0.4208 | 7.5048 |
| CatBoost | 0.9905 | 0.8769 | 0.8382 | 0.8571 | 0.9703 | 0.8731 | 0.0041 | 0.1618 | 0.2732 | 6.0449 |
| Neural Network (MLP) | 0.9805 | 0.7231 | 0.6912 | 0.7068 | 0.9767 | 0.7622 | 0.0093 | 0.3088 | 0.4465 | 8.3654 |

## Experiment 6 — SHAP global importance (deployed model, test sample)

| label | mean_abs_shap | share_pct |
|---|---|---|
| Temp. difference [K] | 0.0181 | 24.8201 |
| Wear x Torque [min·Nm] | 0.0168 | 23.1128 |
| Rotational speed [rpm] | 0.0154 | 21.1524 |
| Power [W] | 0.0093 | 12.7600 |
| Tool wear [min] | 0.0048 | 6.5840 |
| Product type | 0.0033 | 4.5894 |
| Torque [Nm] | 0.0030 | 4.1183 |
| Air temperature [K] | 0.0011 | 1.5126 |
| Temp. diff x Torque [K·Nm] | 0.0006 | 0.8669 |
| Process temperature [K] | 0.0004 | 0.4835 |

## Experiment 7 — error analysis (test)

Recall by failure mode:

| mode | failures_in_test | detected | recall |
|---|---|---|---|
| Tool Wear Failure | 10 | 1 | 0.1000 |
| Heat Dissipation Failure | 29 | 24 | 0.8276 |
| Power Failure | 13 | 13 | 1.0000 |
| Overstrain Failure | 16 | 15 | 0.9375 |
| Random Failure | 0 | 0 | nan |
| No mode label | 2 | 0 | 0.0000 |

Mean feature values by outcome:

| outcome | air_temp_k | process_temp_k | rotational_speed_rpm | torque_nm | tool_wear_min | temp_diff_k | power_w | wear_torque_min_nm | probability | count |
|---|---|---|---|---|---|---|---|---|---|---|
| FN | 300.7410 | 310.2240 | 1500.9410 | 42.1710 | 190.5880 | 9.4820 | 6406.6240 | 7869.0120 | 0.1590 | 17 |
| TN | 299.9770 | 309.9990 | 1538.4550 | 39.5610 | 107.4550 | 10.0220 | 6233.6240 | 4250.8240 | 0.0070 | 1932 |
| TP | 300.7430 | 310.1080 | 1472.4510 | 52.8180 | 140.9020 | 9.3650 | 7533.9460 | 7427.6020 | 0.9410 | 51 |

## Ablation (drop one feature)

| dropped_feature | cv_pr_auc_mean | cv_pr_auc_std | val_pr_auc | delta_val_pr_auc |
|---|---|---|---|---|
| (none — full set) | 0.9203 | 0.0441 | 0.8897 | 0.0000 |
| type | 0.9015 | 0.0634 | 0.8730 | -0.0167 |
| air_temp_k | 0.9188 | 0.0438 | 0.8885 | -0.0011 |
| process_temp_k | 0.9183 | 0.0438 | 0.8923 | 0.0026 |
| rotational_speed_rpm | 0.8605 | 0.0476 | 0.8337 | -0.0560 |
| torque_nm | 0.9192 | 0.0449 | 0.8890 | -0.0007 |
| tool_wear_min | 0.8994 | 0.0438 | 0.8188 | -0.0709 |
| temp_diff_k | 0.8762 | 0.0381 | 0.8351 | -0.0546 |
| power_w | 0.8774 | 0.0529 | 0.8317 | -0.0580 |
| wear_torque_min_nm | 0.8927 | 0.0526 | 0.8632 | -0.0265 |
| temp_diff_x_torque | 0.9176 | 0.0442 | 0.8831 | -0.0066 |

## Optional module — failure-mode classifiers (one-vs-rest, test)

| mode | name | test_positives | test_pr_auc | test_no_skill_pr_auc | test_roc_auc | test_recall | test_precision |
|---|---|---|---|---|---|---|---|
| TWF | Tool Wear Failure | 10 | 0.0331 | 0.0050 | 0.8077 | 0.5000 | 0.0463 |
| HDF | Heat Dissipation Failure | 29 | 1.0000 | 0.0145 | 1.0000 | 0.8966 | 1.0000 |
| PWF | Power Failure | 13 | 0.9822 | 0.0065 | 0.9999 | 0.9231 | 0.9231 |
| OSF | Overstrain Failure | 16 | 0.9883 | 0.0080 | 0.9999 | 1.0000 | 0.8889 |
| RNF | Random Failure | 4 | 0.0042 | 0.0020 | 0.6211 | 0.0000 | 0.0000 |

## Caveats

- AI4I 2020 is a **synthetic** benchmark whose labels follow documented rules; real plants are noisier.
- SHAP shows model attributions, not proven physical causes.
- Test-set sizes are small for failures (~68 cases), so metrics carry sampling uncertainty.