# Constraint guaranteed evasion attacks

[![Build](https://github.com/aucad/new-experiments/actions/workflows/build.yml/badge.svg)](https://github.com/aucad/new-experiments/actions/workflows/build.yml)

This implementation demonstrates an approach to introduce constraints to unconstrained adversarial machine learning evasion attacks.
We develop a constraint validation algorithm, _Contraint Guaranteed Evasion_ (CGE), that guarantees generated evasive adversarial examples satisfy domain constraints.

The experimental setup allows running various adversarial evasion attacks, enhanced with CGE, on different data sets and victim classifiers.
The following options are included.

- **Attacks**: Projected Gradient Descent (PGD), Zeroth-Order Optimization (ZOO), HopSkipJump attack. 
- **Classifiers**: Keras deep neural network and tree-based ensemble XGBoost.
- **Data sets**: Four different data sets, from various domains.
- **Constraints**: Constraints are configurable experiment inputs; configuration files show how to specify them.

**Comparison.** We also include a comparison attack, _Constrained Projected Gradient Descent_ (C-PGD).
It uses a different constraint evaluation approach introduced by [Simonetto et al](https://arxiv.org/abs/2112.01156).

**Data sets**

- [**IoT-23**](https://doi.org/10.5281/zenodo.4743746) - Malicious and benign IoT network traffic; 10,000 rows, 2 classes (sampled).
- [**UNSW-NB15**](https://doi.org/10.1109/MilCIS.2015.7348942) - Network intrusion dataset with 9 attacks; 10,000 rows, 2 classes (sampled). 
- [**URL**](https://doi.org/10.1016/j.engappai.2021.104347) - Legitimate and phishing URLs; 11,430 rows, 2 classes.
- [**LCLD**](https://www.kaggle.com/datasets/wordsforthewise/lending-club) - Kaggle's All Lending Club loan data; 20,000 rows, 2 classes (sampled).

<details>
<summary>Notes on preprocessing and sampling</summary>
<ul>
<li>The input data must be numeric and parse to a numeric type.</li>
<li>Categorical attributes must be one-hot encoded.</li>
<li>Data should not be normalized (otherwise constraints must include manual scaling).</li>
<li>All data sets have 50/50 class label distribution.</li>
<li>The provided sampled data sets were generated by <a href="https://waikato.github.io/weka-blog/posts/2019-01-30-sampling/" target="_blank">random sampling without replacement</a>.</li>
</ul>
</details>

### Repository organization

| Directory    | Description                                       |
|:-------------|:--------------------------------------------------|
| `.github`    | Automated workflows, development instructions     |
| `cge`        | CGE algorithm implementation                      |
| `comparison` | C-PGD attack implementation and its license       |
| `config`     | Experiment configuration files                    |
| `data`       | Preprocessed input data sets                      |
| `exp`        | Source code for running experiments               |
| `plot`       | Utilities for plotting experiment results         |
| `ref_result` | Referential result for comparison                 |
| `test`       | Unit tests                                        |

- The Makefile contains pre-configured commands to ease running experiments.
- The `data/feature_*.csv` files are exclusively for use with C-PGD attack.
- All software dependencies are listed in `requirements.txt`.
  
## Experiment workflow

A single experiment consists of (1) preprocessing and setup (2) training a classification model on a choice data set (3) applying an adversarial attack to that model (4) scoring and (5) recording the result. A constraint-validation approach can be enabled or disabled during the attack to impact the validity of the generated adversarial examples.

<pre>
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐ 
○───┤  args-parser  ├─────┤     setup     ├─────┤      run      ├─────┤      end      ├───◎
    └───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
     inputs:              * preprocess data      k times:                write result
     * data set           * init classifier      1. train model     
     * constraints        * init attack          2. attack
     * other configs      * init validation      3. score
</pre>

## Usage

**Software requirements**

* [Python](https://www.python.org/downloads/) -- version 3.9 or higher
* [GNU make](https://www.gnu.org/software/make/manual/make.html) -- version 3.81 or later

Check your environment using the following command, and install/upgrade as necessary.

```
python3 --version & make --version
```

**Install dependencies**

```
pip install -r requirements.txt --user
```

### Reproducing paper experiments

![](https://img.shields.io/badge/%F0%9F%95%92%2024%E2%80%9448%20hours/each-FFFF00?style=flat-square) **Run attack evaluations.**   
Run experiments for combinations of data sets $\times$ classifiers $\times$ attacks (20 experiment cases). 

<pre>
make attacks   -- run all attacks, using constraint enforcement.
make original  -- run all attacks, but without validation (ignore constraints).
</pre>

![](https://img.shields.io/badge/%F0%9F%95%92%2030%20min%20%E2%80%94%203%20h-FFFF00?style=flat-square) **Run constraint performance test.**   
Uses varying number of constraints to measure performance impact on current hardware. 
This experiment tests PGD, CPGD, VPGD attacks on UNSW-NB15 data set.

```
make perf
```

### Visualizations

**Plots.** Generate plots of experiment results.

```
make plots
```

**Comparison plot.** To plot results from some other directory, e.g. `ref_result`, append a directory name.

```
make plots DIR=ref_result
```

## Extended/Custom usage

The default experiment options are defined statically in `config` files.
An experiment run can be customized further with command line arguments, to override the static options.
To run such custom experiments, call the `exp` module directly.

```
python3 -m exp [PATH] {ARGS}
```

For a list of supported arguments, run:

```
python3 -m exp --help
```

All plotting utilities live separately from experiments in `plot` module.
For plotting help, run:

```
python3 -m plot --help
```
