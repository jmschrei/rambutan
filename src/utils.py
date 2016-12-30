# utils.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines useful utility functions.
"""

import numpy
import pandas

def bedgraph_to_dense(filename):
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

	data = pandas.read_csv(filename, sep="\t", header=None)
	n = data[2][-1]
	array = numpy.zeros(n)

	for _, (_, start, end, value) in data.iterrows():
		array[start:end] = value

	return array

def fasta_to_dense(filename):
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

	with open(filename, 'r') as fasta:
		fasta.readline()

		sequence = ''
		for line in fasta:
			sequence += line.strip('\r\n')

		n = len(sequence)
		array = numpy.zeros((n, 4))

		for i, char in enumerate(sequence):
			if char == 'A':
				array[i, 0] = 1
			elif char == 'C':
				array[i, 1] = 1
			elif char == 'G':
				array[i, 2] = 1
			elif char == 'T':
				array[i, 3] = 1

	return array