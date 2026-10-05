Input data -> Classical preprocessing -> Feature vector -> Encode features into qubits -> Parameterized quantum gates -> Measurement -> Prediction -> Loss -> Classical optimizer -> Update quantum circuit parameters -> Repeat



# Linear Regression
## Data to compare:
| Category | Metrics |
|---|---|
| Accuracy (primary) | Test RMSE and MAE (lower is better), R² |
| Training cost | Total wall-clock time, time per epoch, epochs to best validation epoch |
| Resource use | Peak memory |
| Inference cost | Latency for 1 sample, throughput for a full batch |
| Model | Parameter count, saved file size |
| Reliability | Standard deviation across seeds/folds, number of failed or diverged runs |
## File Structure
### Data
https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
this is where training data can be found
### PrevRun
previous run data should be saved here
### Models
finished models should be saved here