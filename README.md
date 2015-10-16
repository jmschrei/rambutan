# rambutan

caffe has become very popular as a deep learning framework. However, its prototxt files can be hundreds of lines long, and attempting to define models using pycaffe can be frustrating. rambutan is a python wrapper for caffe which aims at providing a simple, pythonic, interface for users so that users can define, train, and evaluate deep models in only a few lines of code. It requires that caffe and pycaffe are both built properly.

# Installation

Installation is as simple as `pip install rambutan` to get the latest released version, or the following to get the bleeding edge:

```
git clone https://github.com/jmschrei/rambutan.git
cd rambutan
python setup.py install
```

# Examples 

