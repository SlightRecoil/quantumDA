Input data -> Classical preprocessing -> Feature vector -> Encode features into qubits -> Parameterized quantum gates -> Measurement -> Prediction -> Loss -> Classical optimizer -> Update quantum circuit parameters -> Repeat

# Installation
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements
```
# Object Detection
## Metrics to be compared
| Category | Metrics |
|---|---|
| Accuracy (primary) | Test mAP@50, mAP@50:95, precision, recall |
| Reliability | Standard deviation across seeds/folds, number of failed or diverged runs |
| Training cost | Total wall-clock time, time per epoch, epochs to best validation epoch |
| Resource use | Peak memory |
| Inference cost | Latency for 1 sample, throughput for a full batch |
| Model | Parameter count, saved file size |


mAP is used as the primary metric because object detection must evaluate both whether an object was detected and whether its predicted bounding box is correct; ordinary classification accuracy does not capture both elements.
