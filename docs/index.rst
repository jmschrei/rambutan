.. rambutan documentation master file, created by
   sphinx-quickstart on Wed Jul  5 19:27:20 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

Rambutan is a package for the prediction of the 3D structure of human cell types. It focuses on the prediction of Hi-C contact maps, but rather than trying to predict the number of contacts that a pair of loci engage in, it instead predicts whether the contact is statistically significant with respect to their genomic distance. This genomic distance effect is extremely important as pairs of loci that are close together are very likely to be in contact simply due to physics as opposed to biological importance, whereas long-range contacts are typically enriched for important biological interactions. The predictions are made using a convolutional neural network that takes in nucleotide sequence and DNaseI sensitivity from two loci spanning 1000 nucleotides. The goal is to serve as a substitute for both running an experiment to collect Hi-C data for a cell type and for running Fit-Hi-C on the data afterwards.

Rambutan solves a big data problem. This means that it is not optimized for use on laptops or old computers, but rather works best on modern computational servers. Most experiments were done using a computer with four K40 GPUs and over 60GB of RAM. To illustrate this, there are over 30 million pairs of positions in chromosome 21, the smallest chromosome, within the band considered by Rambutan. Using all four GPUs, it can easily take approximately an hour for this prediction. While it's certainly possible to run Rambutan even on only the CPU, one must make up for with patience what they lack in hardware.

The manuscript is currently under revisions at Bioinformatics, but the preprint on bioRxiv can be found here: http://www.biorxiv.org/content/early/2017/01/30/103614

Comments and suggestions are always greatly appreciated.

.. toctree::
   :maxdepth: 0
   :hidden:

   self
   installation.rst
   faq.rst
   rambutan.rst
   io.rst
   utils.rst
