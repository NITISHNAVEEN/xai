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

## Reproducing the experiments

The recommended environment is Google Colab with a GPU runtime (the original experiments used a T4). The notebook can also be run locally with Jupyter and a working PyTorch installation. For local CUDA runs, install a PyTorch/torchvision build compatible with your CUDA setup.

### Dependencies

Run the notebook's setup install cell, or run these commands in the environment where Jupyter will use them:

```bash
pip install shap grad-cam captum detectors tqdm
pip install git+https://github.com/marcotcr/lime
```

The notebook also uses PyTorch, torchvision, timm, NumPy, Matplotlib, Pillow, scikit-learn, scikit-image, and OpenCV (`cv2`). Colab includes many of these already; install any missing packages in a local environment. OpenCV is required by the inpainting fidelity strategy.

### Execution steps

1. Open [`notebooks/all-in-one-no-output.ipynb`](./notebooks/all-in-one-no-output.ipynb) in Colab or Jupyter and run the dependency and setup cells near the top.
2. For a fresh run, skip the recovery cells that mount Google Drive and load `/content/drive/MyDrive/vars.pkl`. Instead, uncomment the dataset initialization cell and choose a dataset name supported by `init_dataset` (`cifar10`, `cifar100`, `svhn`, or `imagenet`; the latter loads Imagenette). CIFAR10, CIFAR100, and SVHN are downloaded by the notebook. Prepare Imagenette under `./data` before selecting it.
3. In the corresponding experiment section, select the matching dataset/model preprocessing configuration, choose the explainer, and run its explanation-generation cells to populate `imgs` and `exps`. Keep the same images, model, explainer, and `step` when comparing fidelity strategies. The recovery workflow can be used instead when resuming a run with a previously saved `vars.pkl`.
4. Run each fidelity cell for the selected experiment. They call `calculate_fidelity1` through `calculate_fidelity4`; each returns per-image scores, and `np.mean(scores)` reports the mean for that batch.
5. Repeat for the other dataset, model, or explainer sections you want to evaluate. The notebook fixes random seeds in its setup cells; use the same sample selection and settings when reproducing a comparison.

### Fidelity strategies

All four functions evaluate a deletion curve: pixels are removed in explanation order, the model's probability for the intact image's predicted class is recorded as deletion progresses, and the normalized area under that curve is returned. The replacement applied to removed pixels differs:

| Function | Replacement strategy |
| --- | --- |
| `calculate_fidelity1` | Black pixels |
| `calculate_fidelity2` | Gaussian blur (default `sigma=2`) |
| `calculate_fidelity3` | Dataset mean value (defaults to the image mean when no dataset mean is supplied) |
| `calculate_fidelity4` | OpenCV Telea inpainting (default `radius=2`) |

Use the same inputs and deletion step across all four functions to compare replacement strategies fairly. A lower area under the deletion curve means the selected class probability fell more quickly as the most relevant pixels were removed.
