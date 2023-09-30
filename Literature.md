# Literature



## 0. Overview

Videos:

https://www.youtube.com/watch?v=CIfsB_EYsVI&t=3508s 


Papers:

Adversarial Attacks and Defences: A Survey  (https://arxiv.org/abs/1810.00069)

A Fourier Perspective on Model Robustness in Computer Vision (https://arxiv.org/abs/1906.08988)


## 1. Spectral Adversarial Defense

### Code

https://github.com/bethgelab/foolbox

https://github.com/paulaharder/SpectralAdversarialDefense


### Datasets

 - CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
     - dataset contains 60,000 
     - 32x32 color images in 
     - 10 different classes.


 - CIFAR-100
     - 600 images (500 train, 100 test)
     - 100 classes (20 superclasses)


### Methods to Compare

 - FGSM (2015) - Fast Gradient Sign Method (https://arxiv.org/abs/1412.6572)
   - https://www.youtube.com/watch?v=PFS9KQcQT-s 
   - Notebook: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/adversarial_fgsm.ipynb

 - BIM (2017) - Basic Iterative Method (https://arxiv.org/abs/1607.02533) 
   - https://www.youtube.com/watch?v=RyEhb-KquEY
   - https://www.youtube.com/watch?v=G1FNHgrhEbE
 
 - PGD (2019) - Projected Gradient Descent (https://arxiv.org/abs/1706.06083)
   - https://www.youtube.com/watch?v=zCaiyGeFsgA&t=792s
 
 - Deepfool (2016) (Foolbox - https://foolbox.readthedocs.io/en/v2.4.0)
 
 - C&W (2017) - Carlini & Wagner (https://arxiv.org/abs/1705.07263)
   - https://www.youtube.com/watch?v=yIXNL88JBWQ&t=20s
   - https://www.youtube.com/watch?v=1thoX4c5fFc&t=1017s 



## 2. Laplacian Pooling: Towards Robust CNNs

### Code

Not available

### Datasets

 - CIFAR-10

 - CIFAR-100

 - SVHN - The Street View House Numbers (SVHN) Dataset(http://ufldl.stanford.edu/housenumbers/)
 
 - ImageNet (http://www.image-net.org/)


### Netowrks
 
 - ResNet
 
 
### Methods to Compare

 - FGSM
 
 - BIM 
 
 - PGD
  
 - Deepfool
 
 - C&W 
 
 # My Approach
 
 Goal: Use a bigger dataset, e.g. FFHQ. The higher the resolution, the more difficult.


| Attacks | Nets   | Datasets    | Defenses  |
| ------  | ------  | ------     | ------    |
| PGD     | VGG 16  | MNIST      | LID       |
| FGSM    | ResNet  | CIF-10     | Mahalondis |
| BIM     | AlexNet | CIF-100    | MFS  |
| Deepfool| MobNetv2| ImageNet   | PFS  |
| C&W     | YOLOv3  | CityShapes | ..   |
| JSMA    |         | FaceHQ     |      |
|         |         | CelebA     |      |

CelebA

 ### Datasets
 
  - FFHQ (https://github.com/NVlabs/ffhq-dataset)
    - https://arxiv.org/abs/1812.04948 
    - mainly for GANs
    - 70.000 images
    - different age and ethnicity

  - Others: https://analyticsindiamag.com/10-face-datasets-to-start-facial-recognition-projects/
  - https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121594




# New Papers (these papers need to be checked!)


https://exmediawiki.khm.de/exmediawiki/index.php/Adversarial_Attacks

Both

  - 2020 Attacking and Defending Machine Learning Applications of Public Cloud https://paperswithcode.com/paper/attacking-and-defending-machine-learning


  - https://paperswithcode.com/paper/adversarial-attacks-and-defenses-on-graphs-a

  - 2019 https://paperswithcode.com/paper/the-vulnerabilities-of-graph-convolutional

  - ! 2019 Pytorch toolbox https://paperswithcode.com/paper/advertorch-v01-an-adversarial-robustness

  - 2018 https://paperswithcode.com/paper/ensemble-adversarial-training-attacks-and


  - Cleverhans Lib
    - ! 2015 Limitations https://paperswithcode.com/paper/the-limitations-of-deep-learning-in

    - !  CleverHans Lib https://github.com/cleverhans-lab/cleverhans


  - https://gist.github.com/miwong



Defenses

  - 2021 Provable Defense Against Delusive Poisoning: https://arxiv.org/pdf/2102.04716v1.pdf: hier werden die Daten vergiftet, weil Leute einfach Daten aus dem Internet nehmen...  und darauf eben eine Verteidigung gezeigt. Ist das auch wichtig für uns?

  - 2018 https://paperswithcode.com/paper/provable-defenses-against-adversarial 




Attacks

  - 2021 https://arxiv.org/abs/2002.02196: 
      - https://github.com/Scintillare/AIGan 
      - https://github.com/mathcbc/advGAN_pytorch
      - https://github.com/ctargon/AdvGAN-tf

  - 2020 https://paperswithcode.com/paper/a-little-fog-for-a-large-turn
    - JSMA - 2015 The Limitations of Deep Learning in Adversarial Settings https://arxiv.org/abs/1511.07528 

  - https://paperswithcode.com/paper/cloud-based-image-classification-service-is-1

  - 2016 https://paperswithcode.com/paper/practical-black-box-attacks-against-machine

   - ! 2019 https://paperswithcode.com/paper/advhat-real-world-adversarial-attack-on: ArcFace is the most common public face detection system. The adversarial sticker is prepared with a novel algorithm for off-plane transformations of the image which imitates sticker location on the hat. Such an approach confuses the state-of-the-art public Face ID model LResNet100E-IR, ArcFace@ms1m-refine-v2 and is transferable to other Face ID models.

  - 2019 https://paperswithcode.com/paper/natural-adversarial-examples

  - 2018 https://paperswithcode.com/paper/obfuscated-gradients-give-a-false-sense-of

  - 2018 https://paperswithcode.com/paper/distributionally-adversarial-attack
