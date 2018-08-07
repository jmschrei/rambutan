# Rambutan

[![Travis-CI](https://travis-ci.org/jmschrei/rambutan.svg?branch=master)](https://travis-ci.org/jmschrei/rambutan) [![Documentation Status](https://readthedocs.org/projects/rambutan-py/badge/?version=latest)](http://rambutan-py.readthedocs.io/en/latest/?badge=latest)

Rambutan is a deep convolutional neural network which predicts 3D chromatin architecture using only nucleotide sequence and DNaseI sensitivity. Specifically it predicts whether a pair of 1kb loci engage in a statistically significant contact with respect to the genomic distance effect as defined by Fit-Hi-C. If you've previously used Fit-Hi-C to identify relevant contacts in experimentally acquired Hi-C contact maps, you can now use Rambutan to do that for human cell types which don't have Hi-C data! Rambutan is trained off the deeply sequenced GM12878 experiment from the Rao 2014 paper and so can make predictions at 1kb resolution, far higher than most experimentally acquired contact maps.  

Read the manuscript here! <a href="https://www.biorxiv.org/content/early/2018/07/15/103614">Nucleotide sequence and DNaseI sensitivity are predictive of 3D chromatin architecture</a>

**NOTE: After our original submission we discovered an error in our calling of statistically significant contacts. Briefly, when calculating the prior probability of a contact, we used the number of contacts at a certain genomic distance in a chromosome but divided by the total number of bins in the full genome. When we corrected this mistake we noticed that the Rambutan model, as it curently stands, did not outperform simply using the GM12878 contact map that Rambutan was trained on as the predictor in other cell types. While we investigate these new results, we ask that readers treat this manuscript skeptically.**

The code used to recreate most figures in the paper can be found in Biological_Validation.ipynb. 

## Dependencies

Rambutan is written to be used in Python 2.7, but should work for Python 3 as well. Please open an issue on the issue tracker if this is not the case.

Rambutan requires sklearn, joblib, numpy, progressbar, mxnet, and cython. Of these dependencies, the first four can easily be installed using pip. The last two may be more tricky to get installed due to their efficiency needs. In particular, mxnet is a deep learning package and so requires cuda and cudnn for installation. Please see the <a href="http://mxnet.io/get_started/setup.html">mxnet installation guide</a> for instructions on how to install mxnet. Cython requires a working C++ compiler, which should not be a problem if you are on Ubuntu or a mac (gcc and clang both work well). If you are on a Windows machine you will have to download one. For Python 2 this minimal version of Visual Studio 2008 works well: https://www.microsoft.com/en-us/download/details.aspx?id=44266. For Python 3 this version of the Visual Studio Build Tools has been reported to work: http://go.microsoft.com/fwlink/?LinkId=691126. 

Using the Anaconda distribution may help in the in installation of these dependencies.

## Installation

Rambutan can be installed once all of the dependencies are successfully installed. Currently you can install Rambutan by cloning this repo and installing from source using the following commands:

```
git clone https://github/com/jmschrei/rambutan
cd rambutan
python setup.py install
```

## Usage

Rambutan comes with parameters from a model which has been pre-trained on 12.8 million samples from GM12878 chromosomes 1 through 20. This is the model which is used for all tasks in the manuscript. Making predictions is as simple as calling the predict function:

```
from rambutan import Rambutan
model = Rambutan("path/to/model/file/", iteration=25)
y_pred = model.predict("chr21.fa", "chr21.GM12878.dnase.bedgraph", ctxs=[0, 1, 2, 3])
```

The predict function takes in a filename for a FastA file and a filename for a bedgraph file containing fold change DNaseI values. The context parameter defines which GPUs to use in prediction. The prediction task is parallelized in a manner such that there is a linear speedup with the number of contexts. The resulting matrix of predictions will be sparse, only filling in the upper triangle between the band of 50kb to 1Mb. `min_dist` and `max_dist` can be passed in to the Rambutan object initialization to consider a different band.
