# SemStyle 

Code and models for learning to generate stylized image captions from unaligned text.

[A full list of semstyle results](https://almath123.github.io/semstyle_examples/) on MSCOCO images can be found here. 


A pytorch rewrite of SemStyle is included in "./code".

* This software is written in `python 2.7`
  it assumes that you have installed (or a conda environment containing) 
  `torch, torchvision, scipy`
  
* To setup the semstyle models go to "./code/models/" and run "download.sh".
Then from "./code" run:
```python img_to_text.py --test_folder <folder with your test images>```

  * The `--cpu` flag will disable the gpu.
  * If not already installed, the system will start by downloading inception.v3 image classification model to the local torch directory, this may take a while (https://pytorch.org/docs/stable/torchvision/models.html)

* Training code is included. 

* Scripts to generate the training data from publicly available sources is coming.

### Online demo 

For a limited time (while cpu cycles last): [Live demo, caption your own images](http://43.240.97.39:5000/upload)

A blog post decribing the SemStyle system is here: 
http://cm.cecs.anu.edu.au/post/semstyle/

Citing this work: 

Alexander Mathews, Lexing Xie, Xuming He. **SemStyle: Learning to Generate Stylised Image Captions Using Unaligned Text**, in Conference on Computer Vision and Pattern Recognition (CVPR ‘18), Salt Lake City, USA, 2018.
https://arxiv.org/abs/1805.07030

```
@inproceedings{mathews2018semstyle,
  title={{SemStyle}:  Learning to Generate Stylised Image Captions using Unaligned Text},
  author={Mathews, Alexander and Xie, Lexing and He, Xuming},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```
