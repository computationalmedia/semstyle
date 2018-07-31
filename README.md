# SemStyle 

[Full list of semstyle results](https://almath123.github.io/semstyle_examples/)

Code and models for learning to generate stylized image captions from unaligned text.

A pytorch rewrite of SemStyle is included in "./code".

To setup go to "./code/models/" and run "download.sh".
Then from "./code" run:
```python img_to_text.py --test_folder <folder with your test images>```

The `--cpu` flag will disable the gpu.

Training code is included. Scripts to generated the training data from publicly available sources is coming.
