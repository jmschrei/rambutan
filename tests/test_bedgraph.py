# test_fasta.py
# Author: Jacob Schreiber <jmschr@cs.washington.edu>
# This will test the bedgraph loading tools.

import numpy

from rambutan.utils import bedgraph_to_dense
from rambutan.utils import encode_dnase

from nose.tools import assert_equal
from nose.tools import assert_true
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_equal

def test_bedgraph_shape():
	data = bedgraph_to_dense("tests/test.bedgraph")
	assert_equal(data.ndim, 1)
	assert_equal(data.shape[0], 30)

def test_bedgraph_sum():
	data = bedgraph_to_dense("tests/test.bedgraph")
	assert_almost_equal(data.sum(), 84.6)

def test_zeros_encoding():
	data = numpy.zeros(100)
	encoding = encode_dnase(data)

	assert_equal(encoding.sum(), 100)
	assert_equal(encoding[:, 0].sum(), 0)
	assert_equal(encoding[:, 2].sum(), 100)
	assert_equal(encoding[:, 3].sum(), 0)

def test_ones_encoding():
	data = numpy.ones(100) * numpy.e + 1
	encoding = encode_dnase(data)

	assert_equal(encoding.sum(), 200)
	assert_equal(encoding[:, 0].sum(), 0)
	assert_equal(encoding[:, 2].sum(), 100)
	assert_equal(encoding[:, 3].sum(), 100)

def test_bedgraph_encoding():
	data = bedgraph_to_dense("tests/test.bedgraph")
	encoding = encode_dnase(data)

	assert_equal(encoding.sum(), 59)
	assert_equal(encoding[:, 0].sum(), 3)
	assert_equal(encoding[:, 1].sum(), 8)
	assert_equal(encoding[:, 2].sum(), 30)
	assert_equal(encoding[:, 3].sum(), 9)
	assert_equal(encoding[:, 4].sum(), 9)
	assert_equal(encoding[:, 5].sum(), 0)
	assert_equal(encoding[:, 6].sum(), 0)
