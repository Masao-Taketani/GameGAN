# Learning to Simulate Dynamic Environments with GameGAN

## Framework & Libraries
- torch: 1.7.1
- torchvision: 0.8.2
- torchsummary: 1.5.1
- opencv-contrib-python: 4.5.1.48
- tensorboardX: 2.1
- numpy: 1.19.2
- keyboard: 0.13.5

## Dataset
For this experiment, I used [GTA](https://github.com/Sentdex/GANTheftAuto/tree/main/data/gtav/gtagan_2_sample) dataset.

## Training
```
$ python train.py --datapath [enter your dataset path here]
```
## Test
```
$ python test.py --model_path [enter your model path here]
```

## References
- Papers
  - [Learning to Simulate Dynamic Environments with GameGAN](https://arxiv.org/abs/2005.12126)<br>
  - [Recurrent Environment Simulators](https://arxiv.org/abs/1704.02254)<br>
  - [Neural Turing Machines](https://arxiv.org/abs/1410.5401)<br>
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>
  - [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)<br>
  - [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)<br>
  - [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)<br>
- Repos
  - [Sentdex/GANTheftAuto](https://github.com/Sentdex/GANTheftAuto)<br>
  - [LMescheder/GAN_stability](https://github.com/LMescheder/GAN_stability)<br>
