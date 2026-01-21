# Unifying Graph Out-of-Distribution Generalization and Detection through Spectral Contrastive Invariant learning

[![License][license-image]][license-url]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18327040.svg)](https://doi.org/10.5281/zenodo.18327040)

This is the official code for the implementation of "Unifying Graph Out-of-Distribution Generalization and Detection through Spectral Contrastive Invariant learning"
which is accepted by WWW 2026.

[license-url]: https://github.com/Lowy999/GOODFormer/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg


## Table of contents

* [Installation](#installation)
* [Run](#run)
* [License](#license)
* [Contact](#contact)


## Installation 

### Conda dependencies

```shell
conda env create -f environment.yml
conda activate UniGOOD
```

### Project installation

```shell
pip install -e .
```

## Run

```shell
bash run.sh
```


## License

The UniGOOD codebase is licensed under **GPLv3**:
- Architecture builds upon [GOOD](https://github.com/divelab/GOOD.git) (GPLv3)
- with some code adapted from [Bridge](https://github.com/deeplearning-wisc/graph-spectral-ood) (Apache-2.0 license)
- See full license in [LICENSE](LICENSE)

## Contact

Please feel free to contact [Tianyin Liao](1120230329@mail.nankai.edu.cn)!

