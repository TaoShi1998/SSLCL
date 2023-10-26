# SSLCL: An Efficient Model-Agnostic Supervised Contrastive Learning Framework for Emotion Recognition in Conversations (Under Review by AAAI 2024)


## Introduction
Supervised Sample-Label Contrastive Learning with Soft-HGR Maximal Correlation (SSLCL) is an efficient and model-agnostic supervised contrastive learning framework for the problem of Emotion Recognition in Conversations (ERC), which eliminates the need for a large batch size and can be seamlessly integrated with existing ERC models without introducing any model-specific assumptions. Extensive experiments on two ERC benchmark datasets, IEMOCAP and MELD, demonstrate the compatibility and superiority of our proposed SSLCL framework compared to existing state-of-the-art supervised contrastive learning (SCL) methods. 

The full paper is available at [https://arxiv.org/abs/2310.16676](https://arxiv.org/abs/2310.16676).


## Model Architecture
The overall framework of SSLCL is illustrated as follows, which is made up of three key components: sample feature extraction, label learning, and sample-label contrastive learning. 
![Figure 1: Illustration of the overall framework of SSLCL.](https://github.com/TaoShi1998/SSLCL/assets/37060800/ca59e2f3-46e3-4d4c-85cd-6a79f34152f7)


## Results
The comparisons between SSLCL and existing SCL approaches on IEMOCAP and MELD are shown as follows.
### IEMOCAP
|Model |Happy|Sad|Neutral|Angry|Excited|Frustrated|*Weighted-F1*|
|:----- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|M3Net + CE|52.74|79.39|67.55|69.30|74.39|66.58|69.24|
|M3Net + SupCon|48.80|80.17|66.67|67.68|75.62|66.58|68.86|
|M3Net + mv-SupCon|51.23|80.26|66.17|69.01|69.40|67.25|68.12|
|M3Net + SWFC|54.67|80.85|68.61|67.42|76.92|62.41|69.17|
|M3Net + SSLCL|58.44|82.43|69.32|71.44|77.02|69.34|**71.98**|


### MELD
|Model |Neutral|Surprise|Fear|Sad|Happy|Disgust|Anger|*Weighted-F1*|
|:-----|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|M3Net + CE|79.31|58.76|20.51|40.46|63.21|26.17|52.53|65.47|
|M3Net + SupCon|78.58|59.50|23.19|39.56|65.04|23.16|52.70|65.40|
|M3Net + mv-SupCon|78.11|59.72|23.08|42.05|63.60|23.53|53.91|65.34|
|M3Net + SWFC|78.08|60.31|25.26|39.48|64.12|29.33|53.57|65.42|
|M3Net + SSLCL|79.73|61.03|27.32|42.46|65.08|31.30|54.76|**66.92**|


## Citation
If you find our work helpful to your research, please cite our paper as follows.
```bibtex
@misc{shi2023sslcl,
      title={SSLCL: An Efficient Model-Agnostic Supervised Contrastive Learning Framework for Emotion Recognition in Conversations}, 
      author={Tao Shi and Xiao Liang and Yaoyuan Liang and Xinyi Tong and Shao-Lun Huang},
      year={2023},
      eprint={2310.16676},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


