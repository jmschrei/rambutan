# Rambutan

Rambutan is a deep convolutional neural network which predicts Hi-C contacts. In particular, it takes in nucleotide sequence and DNaseI sensitivity from two loci and predicts whether the pair engages in a significant contact as defined by Fit-Hi-C. If you've previously used Fit-Hi-C to assign statistical confidence estimates to identify relevant contacts in experimentally acquired Hi-C contact maps, you can now use Rambutan to do that for human cell types which don't have Hi-C data! Rambutan is trained off the deeply sequenced GM12878 experiment from the Rao 2014 paper and so can make predictions at 1kb resolution, far higher than most experimentally acquired contact maps.  

## Dependencies

Rambutan is written to be used in Python 2.7, but should work for Python 3 as well. Please open an issue on the issue tracker if this is not the case.

Rambutan requires sklearn, joblib, numpy, progresbar, mxnet, and cython. mxnet is a deep learning package and may have complications in installation due to its need to both connect directly to hardware such as GPUs and have specialized software to improve speed. The installation of both mxnet and cython may be more difficult than a simple pip install. Downloading the Anaconda python distribution will likely solve the cython issue, and the mxnet installation is well documented on their website.

## Installation

Currently you can install Rambutan by cloning this repo and installing from source using the following commands:

```
git clone https://github/com/jmschrei/rambutan
cd rambutan
python setup.py install
```

Since Rambutan does require cython you will need a working C++ compiler. If you are on Ubuntu or Mac this should not be a problem (gcc and clang both work well), but if you are on Windows you may have to download one. For Python 2 this minimal version of Visual Studio 2008 works well: https://www.microsoft.com/en-us/download/details.aspx?id=44266. For Python 3 this version of the Visual Studio Build Tools has been reported to work: http://go.microsoft.com/fwlink/?LinkId=691126

## Usage

Rambutan comes with parameters from a model which has been pre-trained on 12.8 million samples from GM12878 chromosomes 1 through 20. This is the model which is used for all tasks in the manuscript. Making predictions is as simple as calling the predict function:

```
from rambutan import Rambutan
model = Rambutan("path/to/model/file/", iteration=25)
y_pred = model.predict("chr21.fa", "chr21.GM12878.dnase.bedgraph", ctxs=[0, 1, 2, 3])
```

The predict function takes in a filename for a FastA file and a filename for a bedgraph file containing fold change DNaseI values. The context parameter defines which GPUs to use in prediction. The prediction task is parallelized in a manner such that there is a linear speedup with the number of contexts. The resulting matrix of predictions will be sparse, only filling in the upper triangle between the band of 50kb to 1Mb. `min_dist` and `max_dist` can be passed in to the Rambutan object initialization to consider a different band.
