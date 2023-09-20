# Data sets

| Name             | Description                              |  Rows  | Class Distr.  | Inclusion |    DOI     |
|:-----------------|:-----------------------------------------|:------:|:-------------:|:---------:|:----------:|
| [IoT-23][iot]    | Malicious and benign IoT network traffic | 10,000 |  5000 / 5000  |  Sampled  | [🔗][iotd] |
| [UNSW-NB15][uns] | Network intrusion dataset with 9 attacks | 10,000 |  5000 / 5000  |  Sampled  | [🔗][unsd] |
| [URL][url]       | Legitimate and phishing URLs             | 11,430 |  5715 / 5715  |   Full    | [🔗][urld] |
| [LCLD][lcld]     | Kaggle's All Lending Club loan data      | 20,000 | 10000 / 10000 |  Sampled  |    None    | 


**Sampling.** The sampled data sets were generated to obtain equal class distribution using Weka's supervised instance
`SpreadSubsample` ([technical details here](https://waikato.github.io/weka-blog/posts/2019-01-30-sampling/)).

[iot]: https://www.stratosphereips.org/datasets-iot23/
[iotd]: https://doi.org/10.5281/zenodo.4743746
[uns]: https://research.unsw.edu.au/projects/unsw-nb15-dataset
[unsd]: https://doi.org/10.1109/MilCIS.2015.7348942
[url]: https://data.mendeley.com/datasets/c2gw7fy2j4/3
[urld]: https://doi.org/10.1016/j.engappai.2021.104347
[lcld]: https://www.kaggle.com/datasets/wordsforthewise/lending-club