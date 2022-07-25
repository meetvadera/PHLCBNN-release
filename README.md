# Post-hoc Loss Calibration for Bayesian Neural Networks

This repository consists of our code from the paper ["Post-hoc loss-calibration for Bayesian neural networks"](https://arxiv.org/abs/2106.06997) presented at UAI '21. Please refer to the description below to learn more about the usage of each file. All the experimental files use `argparse`, and we provide help text for each of the argument in the python files itself. 


1. `experiment.py`: This file is used to run the code for label corruption experiment. 
2. `sgd_baseline.py`: This file is used to run the SGD baseline for the label corruption experiment. 
3. `experiment_rotated_image.py`: This file is used to run the selective classification experiment. 
4. `experiment_rotated_image_no_student_offline.py`: This file is used to run the selective classification experiment without using an intermediate student model. The student model refered in this file is essentially our post-hoc corrected posterior. 
5. `corrections/laplace_bnn_example.py`, `corrections/sgld_bnn_example.py`, `corrections/variational_bnn_example.py` are the scripts used to run toy data experiments. You'll also have to install the laplace package given in `laplace/` directory.


If you use this repository, please consider citing our paper. The BibTex for our paper is: 

```
@InProceedings{pmlr-v161-vadera21a,
  title = 	 {Post-hoc loss-calibration for Bayesian neural networks},
  author =       {Vadera, Meet P. and Ghosh, Soumya and Ng, Kenney and Marlin, Benjamin M.},
  booktitle = 	 {Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {1403--1412},
  year = 	 {2021},
  editor = 	 {de Campos, Cassio and Maathuis, Marloes H.},
  volume = 	 {161},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {27--30 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v161/vadera21a/vadera21a.pdf},
  url = 	 {https://proceedings.mlr.press/v161/vadera21a.html},
  abstract = 	 {Bayesian decision theory provides an elegant framework for acting optimally under uncertainty when tractable posterior distributions are available. Modern Bayesian models, however, typically involve intractable posteriors that are approximated with, potentially crude, surrogates. This difficulty has engendered loss-calibrated techniques that aim to learn posterior approximations that favor high-utility decisions. In this paper, focusing on Bayesian neural networks, we develop methods for correcting approximate posterior predictive distributions encouraging them to prefer high-utility decisions. In contrast to previous work, our approach is agnostic to the choice of the approximate inference algorithm, allows for efficient test time decision making through amortization, and empirically produces higher quality decisions. We demonstrate the effectiveness of our approach through controlled experiments spanning a diversity of tasks and datasets.}
}

```
