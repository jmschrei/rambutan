# test_fasta.py
# Author: Jacob Schreiber <jmschr@cs.washington.edu>
# This will test the FastA loading tools.

import numpy

from rambutan.utils import fasta_to_dense

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_equal

def test_array_type():
	sequence = fasta_to_dense("tests/test1.fa")
	assert_true(isinstance(sequence, numpy.ndarray))

def test_array_shape():
	sequence = fasta_to_dense("tests/test1.fa")
	assert_equal(sequence.shape, (18, 4))

	sequence = fasta_to_dense("tests/test2.fa")
	assert_equal(sequence.shape, (19, 4))

	sequence = fasta_to_dense("tests/test3.fa")
	assert_equal(sequence.shape, (14, 4))

def test_array_composition():
	sequence = fasta_to_dense("tests/test1.fa")
	assert_array_equal(sequence.sum(axis=0), numpy.array([3, 4, 4, 4]))

	sequence = fasta_to_dense("tests/test2.fa")
	assert_array_equal(sequence.sum(axis=0), numpy.array([0, 0, 0, 0]))

	sequence = fasta_to_dense("tests/test3.fa")
	assert_array_equal(sequence.sum(axis=0), numpy.array([5, 5, 2, 2]))

def test_array_encoding():
	sequence = fasta_to_dense("tests/test1.fa")
	assert_equal(sequence.sum(axis=1).max(), 1)
	assert_equal(sequence.sum(axis=1).min(), 0)

	sequence = fasta_to_dense("tests/test2.fa")
	assert_equal(sequence.sum(axis=1).max(), 0)
	assert_equal(sequence.sum(axis=1).min(), 0)

	sequence = fasta_to_dense("tests/test3.fa")
	assert_equal(sequence.sum(axis=1).max(), 1)
	assert_equal(sequence.sum(axis=1).min(), 1)

def test_sequence_1():
	encoding = numpy.array([[1, 0, 0, 0],
							[0, 1, 0, 0],
							[0, 0, 1, 0],
							[0, 0, 0, 1],
							[0, 0, 1, 0],
							[0, 1, 0, 0],
							[0, 0, 0, 0],
							[0, 0, 0, 0],
							[1, 0, 0, 0],
							[0, 0, 0, 0],
							[0, 1, 0, 0],
							[0, 0, 1, 0],
							[0, 0, 0, 1],
							[0, 0, 0, 1],
							[1, 0, 0, 0],
							[0, 1, 0, 0],
							[0, 0, 1, 0],
							[0, 0, 0, 1]])

	sequence = fasta_to_dense("tests/test1.fa")
	assert_array_equal(sequence, encoding)

	sequence = fasta_to_dense("tests/test2.fa")
	assert_array_equal(sequence, numpy.zeros((19, 4)))

def test_big_composition():
	sequence = fasta_to_dense("tests/chr21.fa")
	assert_array_equal(sequence.sum(axis=0), numpy.array([1746, 784, 872, 1998]))
