# Space-based gravitational wave signal detection and extraction with deep neural network


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge&logo=appveyor)](https://opensource.org/licenses/MIT) [![arXiv](https://img.shields.io/badge/arXiv-2207.07414-red?style=for-the-badge&logo=appveyor)](https://arxiv.org/abs/2207.07414) 

## Introduction

Space-based gravitational wave (GW) detectors will be able to observe signals from sources that are otherwise nearly impossible from current ground-based detection. 
Consequently, the well established signal detection method, matched filtering, will require a complex template bank, leading to a computational cost that is too expensive in practice. 
Here, we develop a high-accuracy GW signal detection and extraction method for all space-based GW sources. 
As a proof of concept, we show that a science-driven and uniform multi-stage deep neural network can identify synthetic signals that are submerged in Gaussian noise. 
Our method has more than 99% accuracy for signal detection of various sources while obtaining at least 95% similarity compared with target signals. 
We further demonstrate the interpretability and strong generalization behavior for several extended scenarios.

## Getting started

Our model is developed based on [SpeechBrain](https://speechbrain.github.io/) toolkit, please install the package via:
  ```
  pip install speechbrain
  ``` 

## Results
The architecture of the model is shown below:
![network|300](images/network.pdf)

## Citation
Please cite the following papers if you find the code useful:
```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```