# Updates

Release 09/22/2020
* Incorporated (a) distributed data parallel trainng and (b) fusedSGD optimizer, resulting in 2x faster training.

# FAZE: Few-Shot Adaptive Gaze Estimation

This repository contains the code for training and evaluation of our ICCV 2019 work, which was presented as an Oral presentation. FAZE is a framework for few-shot adaptation of gaze estimation networks, consisting of equivariance learning (via the **DT-ED** or Disentangling Transforming Encoder-Decoder architecture) and meta-learning with gaze-direction embeddings as input.

![The FAZE Framework](https://ait.ethz.ch/projects/2019/faze/banner.jpg)


## Links
* [NVIDIA Project Page](https://research.nvidia.com/publication/2019-10_Few-Shot-Adaptive-Gaze)
* [ETH Zurich Project Page](https://ait.ethz.ch/projects/2019/faze/)
* [arXiv Page](https://arxiv.org/abs/1905.01941)
* [CVF Open Access PDF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf)
* [ICCV 2019 Presentation](https://conftube.com/video/ByfFufRhuRc?tocitem=17)
* [Pre-processing Code GitHub Repository](https://github.com/swook/faze_preprocess) _(also included as a submodule in this repository)_


## Training and Evaluation

### 1. Datasets

Pre-process the *GazeCapture* and *MPIIGaze* datasets using the code-base at https://github.com/swook/faze_preprocess which is also available as a git submodule at the relative path, `preprocess/`.

If you have already cloned this `few_shot_gaze` repository without pulling the submodules, please run:

    git submodule update --init --recursive

After the dataset preprocessing procedures have been performed, we can move on to the next steps.

### 2. Prerequisites

This codebase should run on most standard Linux systems. We specifically used Ubuntu 

Please install the following prerequisites manually (as well as their dependencies), by following the instructions found below:
* PyTorch 1.3 - https://pytorch.org/get-started/locally/
* NVIDIA Apex - https://github.com/NVIDIA/apex#quick-start 
  * *please note that only NVIDIA Volta and newer architectures can benefit from AMP training via NVIDIA Apex.*

The remaining Python package dependencies can be installed by running:

    pip3 install --user --upgrade -r requirements.txt

### 3. Pre-trained weights for the DT-ED architecture and MAML models

You can obtain a copy of the pre-trained weights for the Disentangling Transforming Encoder-Decoder and for the various MAML models from the following location.

    cd src/
    wget -N https://ait.ethz.ch/projects/2019/faze/downloads/outputs_of_full_train_test_and_plot.zip
    unzip -o outputs_of_full_train_test_and_plot.zip

### 4. Training, Meta-Learning, and Final Evaluation

Run the all-in-one example bash script with:

    cd src/
    bash full_train_test_and_plot.bash

The bash script should be self-explanatory and can be edited to replicate the final FAZE model evaluation procedure, given that hardware requirements are satisfied (8x GPUs, where each are Tesla V100 GPUs with 32GB of memory).

The pre-trained DT-ED weights should be loaded automatically by the script `1_train_dt_ed.py`. Please note that this model can take a long time to train when training from scratch, so we recommend adjusting batch sizes and the using multiple GPUs (the code is multi-GPU-ready).

The Meta-Learning step is also very time consuming, particularly because it must be run for every value of `k` or *number of calibration samples*. The code pertinent to this step is `2_meta_learning.py`, and its execution is recommended to be done in parallel as shown in `full_train_test_and_plot.bash`.

### 5. Outputs

When the full pipeline successfully runs, you will find some outputs in the path `src/outputs_of_full_train_test_and_plot`, in particular:
* **walks/**: mp4 videos of latent space walks in gaze direction and head orientation
* **Zg_OLR1e-03_IN5_ILR1e-05_Net64/**: outputs of the meta-learning step.
* **Zg_OLR1e-03_IN5_ILR1e-05_Net64 MAML MPIIGaze.pdf**: plotted results of the few-shot learning evaluations on MPIIGaze.
* **Zg_OLR1e-03_IN5_ILR1e-05_Net64 MAML GazeCapture (test).pdf**: plotted results of the few-shot learning evaluations on the GazeCapture test set.

## Realtime Demo

We also provide a realtime demo that runs with live input from a webcam in the `demo/` folder. Please check the separate
[demo instructions](https://github.com/NVlabs/few_shot_gaze/blob/master/demo/README.md) for details of 
how to setup and run it.


## Bibtex
Please cite our paper when referencing or using our code.

    @inproceedings{Park2019ICCV,
      author    = {Seonwook Park and Shalini De Mello and Pavlo Molchanov and Umar Iqbal and Otmar Hilliges and Jan Kautz},
      title     = {Few-Shot Adaptive Gaze Estimation},
      year      = {2019},
      booktitle = {International Conference on Computer Vision (ICCV)},
      location  = {Seoul, Korea}
    }


## Acknowledgements
Seonwook Park carried out this work during his internship at NVIDIA. This work was supported in part by the ERC Grant OPTINT (StG-2016-717054).
