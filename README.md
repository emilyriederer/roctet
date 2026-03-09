## `roctet`

The famous [Anscombe's Quartet](https://en.wikipedia.org/wiki/Anscombe%27s_quartet) dataset (and its modern cousin, the [Datasaurus Dozen](https://en.wikipedia.org/wiki/Datasaurus_dozen)) features different datasets with shared summary statistics and regression lines. 
It serves as a cautionary illustration of the importance of EDA. 

`roctet` provides the similar ability to generate numerous datasets consisting of a predictive score and binary target which all have the same AUROC but vary substantially in ROC curve shapes, precision, recall, and otther model evaluation metrics. 

Returned datasets may be useful for teaching purposes or testing the relationship between different model evaluation metrics. 

### Methodology

`roctet` generates a dataset in two steps:

1. Based on an AUROC and control parameters, derive a representative ROC curve
2. Assuming a balanced event rate, back out the implied correct and incorrect predictions implied by the ROC curve.
3. Repeat step (2) while varying the control parameters in step (1)

Step (1) is accomplished by using the CDF of the Beta(a,b) distribution since, like an ROC curve, this is a continuous and monotonic function with a domain and range between 0 and 1. 
The AUC for a CDF represents the expected value. Thus, `AUC = b / (a+b) = (b/a) / (1 + b/a) = r / (1 + r)` where `r = b/a`. 
That is, all curves parameterizes the correct ratio of a and b can recover the desired AUC. 
However, changes in magnitudes of a and b control the spread of the distribution, so varying these can result in curves with materially different shapes. 

Step (2) is accomplished by mapping the CDF into a table of (FPR,TPR) pairs. For each pair, we can calculate the cumulative number of positives and negatives captured as `Cumulate-to-Bin Negatives/Positives Captured = FPR * Total Negatives/Positives Captured`. Bin-wise positive and negatives are calcualted as a lagged difference, and randomly assigned scores within each bin. 

## Usage

To get started, jump in to generate some datasets: 

```python
from roctet import calc_roctet

dfs = calc_roctet(auroc = 0.67, n_sets = 10)
dfs[0].glimpse()
```

## Installation

Install from GitHub:

```bash
python -m pip install "git+https://github.com/emilyriederer/roctet.git"
```

Install a specific release tag:

```bash
python -m pip install "git+https://github.com/emilyriederer/roctet.git@v0.1.0"
```

Developer / editable install:

```bash
git clone https://github.com/emilyriederer/roctet.git
cd roctet
uv sync
uv pip install -e .
```
