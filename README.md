Input data -> Classical preprocessing -> Feature vector -> Encode features into qubits -> Parameterized quantum gates -> Measurement -> Prediction -> Loss -> Classical optimizer -> Update quantum circuit parameters -> Repeat

# Installation
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements
```

# Metrics to be compared (Universal)
| Category | Metrics |
|---|---|
| Training cost | Total wall-clock time, time per epoch, epochs to best validation epoch |
| Resource use | Peak memory |
| Inference cost | Latency for 1 sample, throughput for a full batch |
| Model | Parameter count, saved file size |

# File Structure
| Folder | Purpose |
|---|---|
| Data | Training data should be stored here |
| PrevRun | Results of previous runs should be stored here |
| Models | Finished models should be saved here |

# Linear Regression
## Metrics to be compared
| Category | Metrics |
|---|---|
| Accuracy (primary) | Test RMSE and MAE (lower is better), R² |
| Reliability | Standard deviation across seeds/folds, number of failed or diverged runs |


# Object Detection
## Metrics to be compared
| Category | Metrics |
|---|---|
| Accuracy (primary) | Test mAP@50, mAP@50:95, precision, recall |
| Reliability | Standard deviation across seeds/folds, number of failed or diverged runs |

mAP is used as the primary metric because object detection must evaluate both whether an object was detected and whether its predicted bounding box is correct; ordinary classification accuracy does not capture both elements.

# Object Classification
## Metrics to be compared
| Category | Metrics |
|---|---|
| Accuracy (primary) | Test accuracy, precision |
| Reliability | Deviation across seeds, number of failed runs |
