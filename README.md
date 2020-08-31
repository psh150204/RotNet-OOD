# Self-Supervised Learning for OOD Detection

A Simplified Pytorch implementation of *Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019)*

**The code supports only Multi-class OOD Detection experiment(in-dist: CIFAR-10, Out-of-dist: CIFAR-100/SVHN)** 


- Command 
  - RotNet-OOD
  
    python test.py --method=rot --ood_dataset=cifar100
  
  - baseline
  
    python test.py --method=msp --ood_dataset=svhn

### Results (OOD Detection)
- Metric : AUROC

|                                           | CIFAR-100 |  SVHN  |
|:-----------------------------------------:|:---------:|:------:|
| Maximum Softmax Probability<br>(baseline) |   0.6986  | 0.7190 |
|                   RotNet                  |   0.7931  | 0.9584 |
|           RotNet (rot loss only)          |   0.7132  | 0.9560 |
|        RotNet (KL divergence only)        |   0.7834  | 0.8522 |

### Reference
[1] full code(by authors): https://github.com/hendrycks/ss-ood

[2] Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019): https://arxiv.org/abs/1906.12340

[3] A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks(ICLR 2017): https://arxiv.org/abs/1610.02136

