# Faze: Few-Shot Adaptive Gaze Estimation

This repository will contain the code for training, evaluation, and live demonstration of our ICCV 2019 work, which was presented as an Oral presentation in Seoul, Korea. Faze is a framework for few-shot adaptation of gaze estimation networks, consisting of equivariance learning (via the **DT-ED** or Disentangling Transforming Encoder-Decoder architecture) and meta-learning with gaze embeddings as input.

![The Faze Framework](https://ait.ethz.ch/projects/2019/faze/banner.jpg)

## Setup
Further setup instructions will be made available soon. For now, please pre-process the *GazeCapture* and *MPIIGaze* datasets using the code-base at https://github.com/swook/faze_preprocess

## Additional Resources
* Project Page (ETH Zurich): https://ait.ethz.ch/projects/2019/faze/
* Project Page (Nvidia): https://research.nvidia.com/publication/2019-10_Few-Shot-Adaptive-Gaze
* arXiv Page: https://arxiv.org/abs/1905.01941
* CVF Open Access PDF: http://openaccess.thecvf.com/content_ICCV_2019/papers/Park_Few-Shot_Adaptive_Gaze_Estimation_ICCV_2019_paper.pdf
* Pre-processing Code: https://github.com/swook/faze_preprocess

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

Seonwook Park carried out this work during his internship at Nvidia. This work was supported in part by the ERC Grant OPTINT (StG-2016-717054).
