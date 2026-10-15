Input data -> Classical preprocessing -> Feature vector -> Encode features into qubits -> Parameterized quantum gates -> Measurement -> Prediction -> Loss -> Classical optimizer -> Update quantum circuit parameters -> Repeat

# Installation
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements
```

# Object Classification
## Metrics to be compared
| Category | Metrics |
|---|---|
| Accuracy (primary) | Test accuracy, precision |
| Reliability | Deviation across seeds, number of failed runs |
| Training cost | Total wall-clock time, time per epoch, epochs to best validation epoch |
| Resource use | Peak memory |
| Inference cost | Latency for 1 sample, throughput for a full batch |
| Model | Parameter count, saved file size |
