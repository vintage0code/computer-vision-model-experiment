# README

## Code Usage

### Brief Description

This code experiments with several classic CNN-based networks for the dog task including

classification task and detection task. The author is 王耀斌 (wyb@hhtc.edu.cn), and the

copyright belongs to the author.

Especially, for VGG network, we redesigned vgg16 with the thought of reducing conv block and

grow the output feature to adapt 120 classes. The large full connect layer also be replaced.

For GoogLeNet, we redesign it by reducing the inception module to adapt 120 classes.

And for the detection task, we implement it by the thought of making use of classification-

pretrain model as the backbone and the detection model could go on a regression afterward.

### Requires

<ul>
    <li>Pytorch</li>
    <li>Numpy</li>
    <li>Wandb</li>
    <li>Scipy</li>
    <li>Pytorch_lightning</li>
    <li>Pickle</li>
    <li>OpenTSNE</li>
    <li>Matplotlib</li>
    <li>Sklearn</li>
</ul>

### How to run

There are three models (resnet18，googlenet and vgg) defined in these *.py file. (updating...)

The interface for the entire program is main.py（feature visualization except）.

For classification task, it trains like the following instruction:

```python
python main.py -p vgg10 -rn exp1 -m vgg
```

For detection task, it also by the following instruction but modify a little bit:

```python
python main.py -p googlenet2det -rn exp2 -m google_net2detection
```

or read the command instructions to adjust hyper-parameters.

Good luck!