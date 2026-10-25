# Results

Results on the full FakeMusicCaps dataset (27,605 clips). Notebook `03` compares 12 models on the validation set. Only the top five are then evaluated on the test set.

| Representation + classifier | Validation Macro F1 | Test Macro F1 |
|---|---:|---:|
| MuQ + SVM | 0.986 | 0.990 |
| MuQ + Logistic Regression | 0.980 | 0.984 |
| MERT + SVM | 0.968 | 0.953 |
| MERT + Logistic Regression | 0.963 | 0.956 |
| CLAP + SVM | 0.923 | 0.917 |

Results for all 12 models are in `baseline_full_all_models.csv`.

![Top-5 confusion matrices](figures/top5_confusion_matrices_full.png)

## Files

| File | Contents |
|---|---|
| `baseline_full_all_models.csv` | Validation metrics for all 12 models |
| `baseline_full_top5.csv` | Validation and test metrics for the top five |
| `baseline_full_top5_predictions.csv` | Predictions of the top five (used in notebook `04`) |
| `figures/baseline_comparison_full.png` | Comparison of all 12 models |
| `figures/top5_confusion_matrices_full.png` | Test confusion matrices of the top five |
| `pca_umap_clap_mert_muq_full.png` | PCA and UMAP of CLAP, MERT and MuQ embeddings |

The remaining files are from earlier experiments on a 3,000-clip pilot subset and are not part of the main results.