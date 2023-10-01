# Constrained adversarial attacks

This repository contains an experimental setup for introducing constraints to universal adversarial machine learning attacks.
We introduce a _validation algorithm_ that guarantees generated adversarial examples satisfy domain constraints.

This implementation allows to run adversarial machine learning attacks, with or without constraints,
on multiple data sets, attacks, and classifiers, using our constraint validation algorithm.
The following options are implemented.

- **attacks**: Projected Gradient Descent (PGD), zeroth-order optimization (Zoo), HopSkipJumpAttack, and C-PGD (comparison)
- **classifiers**: Keras Deep neural network and tree-based ensemble XGBoost
- **data**: 4 different data sets (see [descriptions](#data-sets))

PGD, Zoo, and HopSkipJumpAttacks are modified to use our constraint validation algorithm.
The C-PGD attack uses a different constraint-validation approach (cf. [source](https://github.com/serval-uni-lu/moeva2-ijcai22-replication)), and is included to allow comparing its results to the other attacks.
PGD and C-PGD require a neural network classifier.

**Repository organization**

| Directory          | Description                                       |
|:-------------------|:--------------------------------------------------|
| `.github`          | Automated workflows                               |
| `comparison`       | C-PGD attack implementation + its license         |
| `config`           | Experiment configuration files                    |
| `data`             | Experiment data sets                              |
| `exp`              | Experiment runtime and validation implementations |
| `ref_result`       | Referential result for inspection                 |
| `test`             | Implementation unit tests                         |

- The Makefile contains pre-configured commands to ease running experiments.
- The software dependencies are detailed in `requirements.txt`.

## Experiment workflow

<pre>
     ┌───────────┐      ┌────────────┐      ┌────────────┐      ┌────────────┐
○────┤   parse   ├──────┤   setup    ├──────┤    run     ├──────┤    end     ├────◎
     │   args    │      │ experiment │      │ experiment │      │ experiment │
     └───────────┘      └────────────┘      └────────────┘      └────────────┘
      inputs:           preprocess &        k times:             write result
      - data set        init classifier,    1. train model      
      - config incl.    attack, and         2. attack
        constraints     validation          3. score
</pre>

## Usage

Runtime requirements: Python 3.9 or higher.

Install dependencies

```
pip install -r requirements.txt
```

Run experiments for all combinations of data sets, classifiers and attacks.

```
make attacks
```

Generate constraint-dependency graphs for all configurations.

```
make graphs
```

Usage help:

```
python3 -m exp --help
```

<details>
<summary>Development instructions</summary>

First install all dev dependencies:

```
pip install -r requirements-dev.txt
```

Available code quality checks

<pre>
make test    -- Run unit tests
make lint    -- Run linter
make dev     -- Test and lint, all at once
</pre>
</details>


## Data sets

All data sets are preprocessed and located in `data` directory.

| Name                                                               |  Rows  | Inclusion | Link                                                                              |
|:-------------------------------------------------------------------|:------:|:---------:|:----------------------------------------------------------------------------------|
| [**IoT-23**][iot]    <br/>Malicious and benign IoT network traffic | 10,000 |  Sampled  | [10.5281/zenodo.4743746](https://doi.org/10.5281/zenodo.4743746)                  |
| [**UNSW-NB15**][uns] <br/>Network intrusion dataset with 9 attacks | 10,000 |  Sampled  | [10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942)        |
| [**URL**][url]       <br/>Legitimate and phishing URLs             | 11,430 | Complete  | [10.1016/j.engappai.2021.104347](https://doi.org/10.1016/j.engappai.2021.104347)  |
| [**LCLD**][lcld]     <br/>Kaggle's All Lending Club loan data      | 20,000 |  Sampled  | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)            | 


**Sampling.** The sampled data sets were generated to obtain equal class distribution using Weka's supervised instance
[`SpreadSubsample`](https://waikato.github.io/weka-blog/posts/2019-01-30-sampling/).

[iot]: https://www.stratosphereips.org/datasets-iot23
[uns]: https://research.unsw.edu.au/projects/unsw-nb15-dataset
[url]: https://data.mendeley.com/datasets/c2gw7fy2j4/3
[lcld]: https://www.kaggle.com/datasets/wordsforthewise/lending-club