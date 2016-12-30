# utils.pyx
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines useful utility functions.
"""

import numpy
import pandas
import time

from progressbar import ProgressBar

cimport numpy

def fasta_to_dense(filename, bint verbose=True):
	"""Translate the sequence from a file to a one hot encoded dense array.

	Parameters
	----------
	filename : str
		The name of the fasta file to use.

	Returns
	-------
	array : numpy.ndarray, shape=(n, 4)
		A dense array one-hot encoded for a nucleotide. 'N' does not take a
		value.
	"""

	cdef int i, n
	cdef list lines = []
	cdef str sequence
	cdef numpy.ndarray array

	with open(filename, 'r') as fasta:
		fasta.readline()

		for line in fasta:
			lines.append(line[:-1])

		sequence = ''.join(lines).upper()

		n = len(sequence)
		array = numpy.zeros((n, 4))

		if verbose:
			pbar = ProgressBar(maxval=n).start()

		for i in range(n):
			if i % 1000000 == 0:
				if verbose:
					pbar.update(i)

			if sequence[i] == 'A':
				array[i, 0] = 1
			elif sequence[i] == 'C':
				array[i, 1] = 1
			elif sequence[i] == 'G':
				array[i, 2] = 1
			elif sequence[i] == 'T':
				array[i, 3] = 1

	if verbose:
		pbar.finish()
	return array


def bedgraph_to_dense(filename, bint verbose=True):
	"""Read a bedgraph file and return a dense numpy array.

	This will read in an arbitrary bedgraph file and turn it into a dense
	array for faster indexing in the future.

	Parameters
	----------
	filename : str
		The name of the bedgraph file to use.

	Returns
	-------
	array : numpy.ndarray, shape=(n,)
		A dense array of the unpacked values.
	"""

	cdef int i, start, end, n
	cdef double value
	cdef numpy.ndarray array

	data = pandas.read_csv(filename, sep="\t", header=None)
	n = data[2][-1]
	array = numpy.zeros(n)

	if verbose:
		pbar = ProgressBar(maxval=data.shape[0]).start()

	for i, (_, start, end, value) in data.iterrows():
		array[start:end] = value

		if i % 1000 == 0 and verbose:
			pbar.update(i)

	if verbose:
		pbar.finish()
	return array


def encode_dnase(dnase, bint verbose=True):
	"""Take in an array of real DNase values and binary encode the log.

	This transforms the fold change value to the log fold value and then
	encodes this value as a binarization of the rounded log value. This is
	done to balance variance between enrichments and depletions. 
	
	For example, a log fold enrichment of 2.9 would have bits 0 1 2and 3 
	active, whereas a value of -2.2 would have bits 0, -1, and -2 active. 
	This encodes DNase values between -2 and 5, so 8 total bits for each 
	position.

	Parameters
	----------
	dnase : numpy.ndarray, shape=(n,)
		The dnase fold change values read from a bedgraph or bigwig file,
		ranging from near 0 to above.

	Returns
	-------
	encoded_dnase : numpy.ndarray, shape=(n, 8)
		The encoded log fold change values.
	"""

	dnase[dnase == 0] = 1
	dnase = numpy.around(numpy.log(dnase))

	cdef int i, value, n = dnase.shape[0]
	cdef numpy.ndarray encoded_dnase = numpy.zeros((n, 8), dtype='int8')

	if verbose:
		pbar = ProgressBar(maxval=n).start()

	for i in range(n):
		if i % 300000 == 0:
			if verbose:
				pbar.update(i)

		value = dnase[i] + 2
		if value >= 2:
			encoded_dnase[i][2:value+1] = 1
		else:
			encoded_dnase[i][value:3] = 1

	if verbose:
		pbar.finish()
	return encoded_dnase 


def extract_regions(sequence):
	"""Extract the mappable regions for predcitions.

	The mappable regions in this case are defined by those regions which have
	no unmappable ('N') nucleotides in the FASTA file.

	Parameters
	----------
	sequence : numpy.ndarray, shape=(n, 4)
		The one hot encoded sequence numpy array.

	Returns
	-------
	regions : numpy.ndarray, shape=(m,)
		The set of mappable regions (midpoints) from this file.
	"""

	n = sequence.shape[0]
	sums = sequence.sum(axis=1)
	sums = numpy.array([sums[i*1000:(i+1)*1000].sum() for i in range(n)])
	regions = numpy.array([i*1000 + 500 for i in range(n) if sums[i] == 1000])
	return regions

def downsample(numpy.ndarray x, numpy.ndarray regions, int min_dist=50000, \
	int max_dist=1000000):
	"""Downsample a 1kb resolution matrix to a 5kb resolution matrix.

	For each cell in the 5kb resolution matrix, take the maximum probability
	for each cell in the 5x5 grid at the 1kb resolution centered at this point.
	For example, the cell in the 5kb resolution matrix at 2500,2500 will take
	the maximum probability of the cells at 500,500, 500,1500, 500,2500...
	4500,500, 4500,1500... etc. This is equivalent to treating the cells as
	being strongly correlated instead of independent from each other.

	Parameters
	----------
	x : numpy.ndarray, shape=(n, n)
		The 1kb resolution matrix to downsample

	regions : numpy.ndarray, shape=(m,)
		The relevant regions to look at

	min_dist : int, optional
		The minimum distance two regions have to be from each other to be
		considered. Default is 50kb.

	max_dist : int, optional
		The maximum distance two regions can be from each other to be
		considered. Default is 1Mb.

	Returns
	-------
	y : numpy.ndarray, shape=(n/5, n/5)
		The 5kb resolution matrix produced from the 1kb matrix.
	"""

	cdef int mid1, mid2, i, j, k1, k2, l1, l2
	cdef numpy.ndarray y = numpy.zeros((9626, 9626))

	for mid1 in regions:
		for mid2 in regions:
			if min_dist <= mid2 - mid1 <= max_dist:
				k1 = (mid1 - 500) / 1000
				k2 = (mid2 - 500) / 1000
				l1 = (mid1 - 2500) / 5000
				l2 = (mid2 - 2500) / 5000

				for i in range(-2, 3):
					for j in range(-2, 3):
						y[l1, l2] = max(y[l1, l2], x[k1 + i, k2 + j])

	return y

def insulation_score(numpy.ndarray x, int size=200):
	"""Calculate the insulation score for a given matrix of any resolution.

	This will slide a size*size square along the diagonal of the matrix, 
	summing the values in the upper triangle of the matrix. If a region
	has no contacts it will not have an insulation score, which handles both
	edge cases.

	Parameters
	----------
	x : numpy.ndarray, shape=(n, n)
		The matrix to calculate the insulation score on. Either Rambutan
		predictions or contacts from a Hi-C map.

	size : int, optional
		The size of the square, default is 200, which is 1Mb on a 5kb
		resolution map.

	Returns
	-------
	insulation : numpy.ndarray, shape=(n,)
	"""

	cdef int i, j, k, n = x.shape[0]
	cdef numpy.ndarray insulation = numpy.zeros(n)
	cdef numpy.ndarray sums = x.sum(axis=0)

	for i in range(n):
		if sums[i] > 0:
			for j in range(-size/2, (size/2)+1):
				if i+j >= n:
					break

				for k in range(j, (size/2)+1):
					if i+k >= n:
						break

					insulation[i] += x[i+j, i+k]

	return insulation