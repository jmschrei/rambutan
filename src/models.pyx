# models.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

"""
This file defines a MxNet model, and all code related to training or predicting
using a MxNet model.
"""

import logging
import mxnet as mx

from mxnet.symbol import Pooling, Variable, Flatten, Concat
from mxnet.symbol import SoftmaxOutput, FullyConnected, Dropout
from mxnet.io import *
from mxnet.ndarray import array

mx.random.seed(0)
random.seed(0)

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
	seq = Convolution(seq, 48, (7, 4), pad=(3, 0))
	seq = Pooling(seq, kernel=(3, 1), stride=(3, 1), pool_type='max')
	seq = Convolution(seq, 48, (7, 1), pad=(3, 0))
	seq = Pooling(seq, kernel=(3, 1), stride=(3, 1), pool_type='max')

	dnase = Pooling(dnase, kernel=(9, 1), stride=(9, 1), pool_type='max')
	dnase = Convolution(dnase, 8, (1, 8))

	x = Concat(seq, dnase)
	x = Convolution(x, 64, (3, 1))
	x = Convolution(x, 64, (3, 1))
	x = Flatten(Pooling(x, kernel=(107, 1), stride=(107, 1), pool_type='max'))
	x = Dense(x, 256)
	return x

def RambutanSymbol(**kwargs):
	x1seq = Variable(name="x1seq")
	x1dnase = Variable(name="x1dnase")
	x1 = Arm(x1seq, x1dnase)

	x2seq = Variable(name="x2seq")
	x2dnase = Variable(name="x2dnase")
	x2 = Arm(x2seq, x2dnase)

	xd = Variable(name="distance")
	xd = Dense(xd, 64)

	x = Concat(x1, x2, xd)
	x = Dense(x, 256)
	x = mx.symbol.FullyConnected(x, num_hidden=2)
	y = SoftmaxOutput(data=x, name='softmax')
	model = mx.model.FeedForward(symbol=y, **kwargs)
	return model

class Rambutan(object):
	"""A Rambutan significant contact predictor.

	Do stuff.
	"""

	def __init__(self, name=None, iteration=None, learning_rate=0.01, 
		num_epoch=25, epoch_size=500, wd=0.0, optimizer='adam', batch_size=1024,
		min_dist=50000, max_dist=1000000, use_seq=True, use_dnase=True,
		use_dist=True, verbose=True):

		self.name = name
		self.iteration = iteration
		self.learning_rate = learning_rate
		self.num_epoch = num_epoch
		self.epoch_size = epoch_size
		self.wd = wd
		self.optimizer = optimizer
		self.batch_size = batch_size
		self.min_dist = min_dist
		self.max_dist = max_dist
		self.use_seq = use_seq
		self.use_dnase = use_dnase
		self.use_dist = use_dist
		self.verbose = verbose

	def predict(self, sequence, dnase, regions, ctx=[0]):
		"""Make predictions and shit.
		"""

		Parallel(n_jobs=len(ctx))( delayed(predict_task)(name, iteration, ctx, 
			len(ctx), sequence, dnase, regions, self.use_seq, self.use_dnase, 
			self.use_dist, self.min_dist, self.max_dist, self.batch_size,
			self.verbose) for ctx in ctxs)

		n = regions.max() / 1000
		y = numpy.zeros((n, n))

		for ctx in ctxs:
			with open('.rambutan.predictions.{}.txt'.format(ctx), 'r') as infile:
				for line in infile:
					mid1, mid2, p = line.split()
					mid1 = (int(float(mid1)) - 500) / 1000
					mid2 = (int(float(mid2)) - 500) / 1000
					p = float(p)
					y[mid1, mid2] = p

			os.system('rm .rambutan.predictions.{}.txt'.format(ctx))

		return y

	def fit(self, sequence, dnase, contacts, regions, validation_sequence=None, 
		validation_dnase=None, validation_contacts=None, validation_regions=None,
		ctxs=[0], eval_metric=roc_auc_score):

		model = RambutanSymbol(ctx=map(mx.gpu, ctxs),
			                   epoch_size=self.epoch_size,
			                   num_epoch=self.num_epoch,
			                   learning_rate=self.learning_rate,
			                   wd=self.wd,
			                   optimizer=self.optimizer
			    )	

		validation = (validation_sequence is not None and
					  validation_dnase is not None and
					  validation_contacts is not None and
					  validation_regions is not None)

		X_train = TrainingGenerator(sequence, dnase, contacts, regions,
			self.batch_size, min_dist=self.min_dist, max_dist=self.max_dist,
			use_seq=self.use_seq, use_dnase=self.use_dnase, 
			use_dist=self.use_dist)

		if validation:
			X_validation = ValidationGenerator(validation_sequence, 
				validation_dnase, validation_contacts, validation_regions,
				batch_size=self.batch_size, min_dist=self.min_dist, 
				max_dist=self.max_dist, use_seq=self.use_seq, 
				use_dnase=self.use_dnase, use_dist=self.use_dist)

		model.fit( X=X_train,
				   eval_data=X_validation if validation else None,
				   eval_metric=roc_auc_score,
				   batch_end_callback=mx.callback.Speedometer(self.batch_size),
				   kvstore='device',
				   epoch_end_callback=mx.callback.do_checkpoint("rambutan")
		)

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
		print "GPU [{}] -- model loaded".format(ctx)

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
							 'distance' : numpy.zeros((10240, 191)) }

				if k != 10240:
					if use_seq:
						data['x1seq'][k] = sequence[mid1-width:mid1+width]
						data['x2seq'][k] = sequence[mid2-width:mid2+width]

					if use_dnase:
						data['x1dnase'][k] = dnase[mid1-width:mid1+width]
						data['x2dnase'][k] = dnase[mid2-width:mid2+width]

					if use_dist:
						distance = mid2 - mid1 - min_dist
						for l in range(100):
							data['distance'][k][l] = 1 if distance >= l*1000 else 0
						for l in range(91):
							data['distance'][k][l+100] = 1 if distance >= 100000 + l*10000 else 0

					predictions[k, 0] = mid1
					predictions[k, 1] = mid2

					k += 1
					tot += 1

				else:
					if verbose:
						print "GPU [{}] -- {} samples loaded, predicting...".format(ctx, k),
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
						print
						print "GPU [{}] -- {} samples predicted and output".format(ctx, tot)