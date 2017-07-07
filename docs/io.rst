.. _io:

Data Generators
===============

Rambutan uses two data generators, the training generator and the validation generator. Both take in regions of the genome, both one-hot encoded nucleotide sequence and bit encoded DNaseI sequence, and output a random sample of pairs of regions for the Rambutan model. Essentially, minibatches are created on the fly from 1D genome data because the nucleotide level input for all pairs in the genome cannot possibly fit in memory. The major difference between the two is that the training generator randomly produces minibatches over all chromosomes that it is fed, whereas the validation generator will systematically yield all positive samples once with an equal number of negative samples. This allows an entire chromosome to be used as a validation set while not double counting regions.

API Reference
-------------

.. automodule:: rambutan.io
	:members:
	:inherited-members:
