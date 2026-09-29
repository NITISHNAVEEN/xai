# A Comparative Evaluation of Explainability Techniques for Image Data

This repository includes the Jupyter notebooks used to evaluate different explainability techniques, across five metrics representing different aspects of quality, specifically _fidelity_, _stability_, _identity_, _separability_ and _time_.

The techniques covered are LIME ([repo](https://github.com/marcotcr/lime)), SHAP ([repo](https://github.com/shap/shap)), GradCAM and GradCAM++ ([repo](https://github.com/jacobgil/pytorch-grad-cam)), IntGrad and SmoothGrad ([repo](https://github.com/pytorch/captum)).

Three common benchmarking datasets were used in the experiments: CIFAR10, SVHN and Imagenette. All datasets were sourced from [PyTorch](https://github.com/pytorch/pytorch).

For each dataset, three models with different architectures were used: VGG16BN, ResNet50, Densenet151. Pretrained models for CIFAR10 and SVHN were sourced from [detectors](https://github.com/edadaltocg/detectors/) library, while models for Imagenet were sourced from [PyTorch](https://github.com/pytorch/pytorch).

All experiments were originally performed in Google Colab using T4 instance to ensure access to CUDA, in a single notebook. To ensure completeness, and since the output visualisations are quite large, for Github the original notebook was split. Each split contains the **same** definitions for initializing datasets and models, definitions for metrics, and definitions for explainer adapters, and can be run intependently, including in Google Colab. Contents of each file is explained below:
- [all-in-one-no-output](/notebooks/all-in-one-no-output.ipynb) - the original notebook with the code for the setup, metrics and all experiments, but **without output** (~275kb).
- [all-in-one-cifar10](/notebooks/all-in-one-cifar10.ipynb) - contains full results only for CIFAR10 (~20mb).
- [all-in-one-svhn](/notebooks/all-in-one-svhn.ipynb) - notebook contains full results only for SVHN (~20mb).
- [all-in-one-imagenette-vgg16bn](/notebooks/all-in-one-imagenette-vgg16bn.ipynb) - contains full results only for Imagenette using VGG16 (~60mb).
- [all-in-one-imagenette-resnet50](/notebooks/all-in-one-imagenette-resnet50.ipynb) - contains full results only for Imagenette using ResNet50 (~60mb).
- [all-in-one-imagenette-densenet121](/notebooks/all-in-one-imagenette-densenet121.ipynb) - contains full results only for Imagenette using Densenet121 (~60mb).
- [all-in-one-imagenette-densenet121](/notebooks/all-in-one-other.ipynb) - contains results for miscelanous experiments (examples for the paper, measuring model accuracy etc.)
