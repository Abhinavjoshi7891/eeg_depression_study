# 📊 Analysis Report: Approach 1 (LOSO)

**Date:** 2026-02-04 10:13
**Total Subjects:** 40 (40)

## 1. Executive Summary
| Metric | Mean | Std Dev | Description |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.690** | 0.231 | Overall correctness |
| Precision | 0.650 | 0.483 | False Positive rate proxy |
| Recall | 0.507 | 0.398 | Ability to find MDD |
| F1 Score | 0.562 | 0.433 | Balance of Prec/Rec |

## 2. Per-Class Performance
- **Healthy Recall (Specificity):** `52.3%` (Ability to identify healthy people)
- **MDD Recall (Sensitivity):** `78.1%` (Ability to detect depression)

## 3. High Variance Subjects
Subjects where the model struggled significantly (Accuracy < 50%):
| test_subject   |   accuracy |
|:---------------|-----------:|
| H_14           |  0.428973  |
| H_16           |  0.13519   |
| H_24           |  0.112047  |
| H_27           |  0.0747986 |
| H_4            |  0.44842   |
| MDD_19         |  0.377856  |
| MDD_5          |  0.154574  |

