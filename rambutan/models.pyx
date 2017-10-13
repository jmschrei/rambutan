# models.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines a MxNet model, and all code related to training or predicting
using a MxNet model.
"""

import numpy, logging
cimport numpy

try:
	import mxnet as mx
	from mxnet.symbol import Pooling, Variable, Flatten, Concat
	from mxnet.symbol import SoftmaxOutput, FullyConnected
	mx.random.seed(0)
except (OSError, ImportError) as e:
	print("Warning: mxnet not properly imported with message {}".format(e.args[0]))
	mx = object
	Pooling, Variable, Flatten, Concat = object, object, object, object
	SoftmaxOutput, FullyConnected = object, object

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def Convolution(x, num_filter, kernel, stride=(1, 1), pad=(0, 0)):
	"""Create a convolution layer with batch normalization and relu activations."""

	x = mx.symbol.Convolution(data=x, num_filter=num_filter, kernel=kernel, 
		stride=stride, pad=pad, cudnn_tune='fastest')
	x = mx.symbol.BatchNorm(data=x)
	x = mx.symbol.Activation(data=x, act_type='relu')
	return x


def Dense(x, num_hidden):
	"""Create an inner product layer with ReLU activations."""

	x = FullyConnected(data=x, num_hidden=num_hidden)
	x = mx.symbol.BatchNorm(data=x)
	x = mx.symbol.Activation(data=x, act_type='relu')
	return x


def Arm(seq, dnase):
	seq = Convolution(seq, 96, (7, 4), pad=(3, 0))
	seq = Pooling(seq, kernel=(3, 1), stride=(3, 1), pool_type='max')
	seq = Convolution(seq, 96, (7, 1), pad=(3, 0))
	seq = Pooling(seq, kernel=(3, 1), stride=(3, 1), pool_type='max')

	dnase = Pooling(dnase, kernel=(9, 1), stride=(9, 1), pool_type='max')
	dnase = Convolution(dnase, 32, (1, 8))

	x = Concat(seq, dnase)
	x = Convolution(x, 128, (3, 1))
	x = Convolution(x, 128, (3, 1))
	x = Flatten(Pooling(x, kernel=(555, 1), stride=(555, 1), pool_type='max'))
	x = Dense(x, 128)
	return x


def RambutanSymbol(**kwargs):
	x1seq = Variable(name="x1seq")
	x1dnase = Variable(name="x1dnase")
	x1 = Arm(x1seq, x1dnase)

	x2seq = Variable(name="x2seq")
	x2dnase = Variable(name="x2dnase")
	x2 = Arm(x2seq, x2dnase)

	xd = Variable(name="distance")
	xd = Dense(xd, 32)

	x = Concat(x1, x2, xd)
	x = Dense(x, 256)
	x = mx.symbol.FullyConnected(x, num_hidden=2)
	y = SoftmaxOutput(data=x, name='softmax')
	model = mx.model.FeedForward(symbol=y, **kwargs)
	return model


def predict_task(name, iteration, ctx, n_jobs, numpy.ndarray sequence, 
	numpy.ndarray dnase, numpy.ndarray regions, bint use_seq=True, 
	bint use_dnase=True, bint use_dist=True, int min_dist=50000, 
	int max_dist=1000000, batch_size=1024, bint verbose=False):
	cdef int k = 0, tot = 0, i, j, l, mid1, mid2
	cdef numpy.ndarray predictions = numpy.zeros((10240, 3), dtype='float32')
	cdef int n = regions.shape[0]

	model = mx.model.FeedForward.load(name, iteration, ctx=mx.gpu(ctx))
	
	if verbose:
		print("GPU [{}] -- model loaded".format(ctx))

	with open('.rambutan.predictions.{}.txt'.format(ctx), 'w') as outfile:
		for mid1 in regions:
			for mid2 in regions[ctx::n_jobs]:
				if not min_dist <= mid2 - mid1 <= max_dist:
					continue

				if k == 0:
					data = { 'x1seq'    : numpy.zeros((10240, 1000, 4)),
							 'x2seq'    : numpy.zeros((10240, 1000, 4)),
							 'x1dnase'  : numpy.zeros((10240, 1000, 8)),
							 'x2dnase'  : numpy.zeros((10240, 1000, 8)),
							 'distance' : numpy.zeros((10240, 100)) }

				if k != 10240:
					if use_seq:
						data['x1seq'][k] = sequence[mid1-500:mid1+500]
						data['x2seq'][k] = sequence[mid2-500:mid2+500]

					if use_dnase:
						data['x1dnase'][k] = dnase[mid1-500:mid1+500]
						data['x2dnase'][k] = dnase[mid2-500:mid2+500]

					if use_dist:
						distance = mid2 - mid1
						for i in range(50):
							data['distance'][k][i] = 0 if distance >= 50000 + i*1000 else 1
						for i in range(40):
							data['distance'][k][i+50] = 0 if distance >= 100000 + i*10000 else 1
						for i in range(10):
							data['distance'][k][i+90] = 0 if distance >= 500000 + i*100000 else 1

					predictions[k, 0] = mid1
					predictions[k, 1] = mid2

					k += 1
					tot += 1

				else:
					data['x1seq'] = data['x1seq'].reshape((10240, 1, 1000, 4))
					data['x2seq'] = data['x2seq'].reshape((10240, 1, 1000, 4))
					data['x1dnase'] = data['x1dnase'].reshape((10240, 1, 1000, 8))
					data['x2dnase'] = data['x2dnase'].reshape((10240, 1, 1000, 8))

					X = mx.io.NDArrayIter(data, batch_size=batch_size)
					y = model.predict(X)
					k = 0

					data['x1seq'] = data['x1seq'].reshape((10240, 1000, 4))
					data['x2seq'] = data['x2seq'].reshape((10240, 1000, 4))
					data['x1dnase'] = data['x1dnase'].reshape((10240, 1000, 8))
					data['x2dnase'] = data['x2dnase'].reshape((10240, 1000, 8))

					predictions[:,2] = y[:,1]
					for mid1, mid2, y in predictions:
						outfile.write( "{} {} {}\n".format(mid1, mid2, y) )

					predictions *= 0

					if verbose:
						print("GPU [{}] -- {} samples predicted and output".format(ctx, tot))

		if verbose:
			print("GPU [{}] -- {} samples loaded, predicting...".format(ctx, k))

		data['x1seq'] = data['x1seq'].reshape((10240, 1, 1000, 4))[:k]
		data['x2seq'] = data['x2seq'].reshape((10240, 1, 1000, 4))[:k]
		data['x1dnase'] = data['x1dnase'].reshape((10240, 1, 1000, 8))[:k]
		data['x2dnase'] = data['x2dnase'].reshape((10240, 1, 1000, 8))[:k]
		data['distance'] = data['distance'][:k]

		X = mx.io.NDArrayIter(data, batch_size=batch_size if batch_size <= k else k)
		y = model.predict(X)

		predictions[:k,2] = y[:,1]
		for mid1, mid2, y in predictions[:k]:
			outfile.write( "{} {} {}\n".format(mid1, mid2, y) )

		if verbose:
			print("\nGPU [{}] -- {} samples predicted and output".format(ctx, tot))