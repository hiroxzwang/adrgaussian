# AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius 

## SIGGRAPH Asia 2024

### [Project Page](https://hiroxzwang.github.io/publications/adrgaussian/) | [Paper](https://arxiv.org/pdf/2409.08669) | [Supplementary Material](https://hiroxzwang.github.io/publications/adrgaussian/static/pdf/saconferencepapers24-118-appendix.pdf)

[AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius](https://hiroxzwang.github.io/publications/adrgaussian/)<br>
[Xinzhe Wang<sup>*</sup>](https://hiroxzwang.github.io/), [Ran Yi<sup>*</sup>](https://yiranran.github.io/), [Lizhuang Ma<sup>†</sup>](https://dmcv.sjtu.edu.cn/people/)<br>
<sup>*</sup>Equal contribution. &emsp; <sup>†</sup>Corresponding author.

![Teaser image](assets/fig-teaser.jpg)

Official implementation of the paper "AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius"

<a href="https://dmcv.sjtu.edu.cn/"><img height="50" src="assets/logo-dmcv.png"> </a>

## 📺 Fast Forward
<details open class="de">
<summary>
<span style="font-weight: bold;">fast-forward.mp4</span>
</summary>
<video controls width="100%">
  <source src="assets/fast-forward.mp4" type="video/mp4">
</video>
</details>

3D Gaussian Splatting enables real-time rendering of complex scenes but still remains unnecessary overhead. We propose AdR-Gaussian, which employs lossless early culling to narrow the tile range of each Gaussian, and proposes a load balancing method to minimize thread waiting time, achieving significant acceleration in rendering speed.

## 🔭Overview

The codebase is built off of the codebase of [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
using PyTorch and CUDA extensions in a Python environment to produce trained models with similar requirements.
You can easily get started using the commands of the seminal project.

### 📕Cloning the Repository
The repository contains submodules.
```shell
# SSH
git clone git@github.com:hiroxzwang/adrgaussian.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/hiroxzwang/adrgaussian.git --recursive
```

### 🚀Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate adr_gaussian
```

### 👈Training

To run the optimizer, simply use

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to save model>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --model_path / -m 
  Path where the trained model should be stored (```output/<random>``` by default).
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**
  #### --data_device
  Specifies where to put the source image data, ```cuda``` by default, recommended to use ```cpu``` if training on large/high-resolution dataset, will reduce VRAM consumption, but slightly slow down training. Thanks to [HrsPythonix](https://github.com/HrsPythonix).
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --convert_SHs_python
  Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.
  #### --debug
  Enables debug mode if you experience erros. If the rasterizer fails, a ```dump``` file is created that you may forward to us in an issue so we can take a look.
  #### --debug_from
  Debugging is **slow**. You may specify an iteration (starting from 0) after which the above debugging becomes active.
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --ip
  IP to start GUI server on, ```127.0.0.1``` by default.
  #### --port 
  Port to use for GUI server, ```6009``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.

</details>
<br>

### 🔍Evaluation
By default, the trained models use all available images in the dataset. To train them while withholding a test set for evaluation, use the ```--eval``` flag. This way, you can render training/test sets and produce error metrics as follows:
```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> --eval -m <path to save the trained model> # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --measure_fps
  Flag to measure the rendering speed.
  #### --skip_render
  Flag to skip writing the image files.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --resolution / -r
  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --convert_SHs_python
  Flag to make pipeline render with computed SHs from PyTorch instead of ours.
  #### --convert_cov3D_python
  Flag to make pipeline render with computed 3D covariance from PyTorch instead of ours.

</details>

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

## 🙇‍Funding and Acknowledgments

This work was supported by National Natural Science Foundation of China (No. 72192821, 62302296, 62302297, 62272447), Shanghai Municipal Science and Technology Major Project (2021SHZDZX0102), Shanghai Sailing Program (22YF1420300), Young Elite Scientists Sponsorship Program by CAST (2022QNRC001), the Fundamental Research Funds for the Central Universities (project number: YG2023QNA35, YG2023QNB17, YG2024QNA44).

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">🤝BibTeX</h2>
If you find this code helpful for your research, please cite:
    <pre><code>@inproceedings{xzwang2024adrgaussian,
  title={AdR-Gaussian: Accelerating Gaussian Splatting with Adaptive Radius},
  author={Wang, Xinzhe and Yi, Ran and Ma, Lizhuang},
  booktitle={ACM SIGGRAPH Asia 2024 Conference Proceedings},
  pages={1--10},
  year={2024}
}</code></pre>
  </div>
</section>
