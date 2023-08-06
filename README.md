This repository contains an implementation of C-MF-VI, the algorithm from paper "Collapsed Variational Bounds for Bayesian Neural Networks", published at Neurips conference 2021.

Link to the [paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/d5b03d3acb580879f82271ab4885ee5e-Paper.pdf).

Requires torch >= 2.0

To run vectorized MNIST experiment:

```
python vmnist.py 
```

If you find our code or paper helpful, please cite:
```
@inproceedings{NEURIPS2021_d5b03d3a,
 author = {Tomczak, Marcin and Swaroop, Siddharth and Foong, Andrew and Turner, Richard},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {25412--25426},
 publisher = {Curran Associates, Inc.},
 title = {Collapsed Variational Bounds for Bayesian Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/d5b03d3acb580879f82271ab4885ee5e-Paper.pdf},
 volume = {34},
 year = {2021}}
```