# rambutan.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

import os, numpy

from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from .io import TrainingGenerator, ValidationGenerator
from .utils import bedgraph_to_dense, fasta_to_dense
from .utils import encode_dnase, extract_regions

class Rambutan(object):
	"""Rambutan: a predictor of mid-range DNA-DNA contacts.

	This serves as a wrapper for all functionality involving the use of Rambutan.
	There are two main functions to use, fit and predict. Fit involves taking in
	nucleotide sequence, DNaseI sensitivity, and a contact map, and training the
	model. Predict involves taking in nucleotide sequence and DNaseI sensitivity
	and predicting significant contacts.

	Note: Due to a limitation of mxnets part, you cannot fit and predict in the
	same program. You must fit the model and save the parameters during training,
	and then load the pre-fit model and make predictions.

	Parameters
	----------
	name : str, optional
		The name of the model, necessary for saving or loading parameters.
		Default is 'rambutan'.

	iteration : int or None, optional
		The iteration of training to load model parameters from, if using Rambutan
		in predict mode. Default is None.

	learning_rate : float, optional
		The learning rate for the optimizer. Default is 0.01.

	num_epoch : int, optional
		The number of epochs to train the model for. Default is 25.

	epoch_size : int, optional
		The number of batches which comprise an 'epoch'. Default is 500.

	wd : float, optional
		The weight decay. This is equivalent to L2 regularization. Default is
		0.0.

	optimizer : str, optional
		The optimizer to use for training. Default is 'adam'.

	batch_size : int, optional
		The number of samples to use in each batch. Default is 1024.

	min_dist : int, optional
		The minimum distance to consider contacts for. Default is 50kb.

	max_dist : int, optional
		The maximum distance to consider contacts for. Default is 1mb.

	use_seq : bool, optional
		Whether to use nucleotide sequence as an input to the model in the 
		training step. Default is True.

	use_dnase : bool, optional
		Whether to use DNaseI sensitivity as an input to the model in the
		training step. Default is True.

	use_dist : bool, optional
		Whether to use genomic distance as an input to the model in the
		training step. Default is True.

	verbose : bool, optional
		Whether to output information during training and prediction. Default
		is True.

	Example 
	-------
	>>> from rambutan import Rambutan
	>>> import numpy
	>>> y_pred = Rambutan(iteration=25).predict('chr21.fa', 'chr21.GM12878.dnase.bedgraph', ctxs=[0, 1, 2, 3])
	>>> numpy.save("chr21.predictions.npy", y_pred)
	"""

	def __init__(self, name='rambutan', iteration=None, learning_rate=0.01, 
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

	def predict(self, sequence, dnase, regions=None, ctxs=[0]):
		"""Make predictions and return the matrix of probabilities.

		Rambutan will make a prediction for each pair of genomic loci defined in
		`regions' which fall between `min_dist' and `max_dist'. Inputs can either
		be appropriately encoded sequence and dnase files, or fasta files and
		bedgraph files for the nucleotide sequence and DNaseI sensitivity
		respectively. Note: fasta files and bedgraph files must be made up of
		a single chromosome, not one entry per chromosome.

		Parameters
		----------
		sequence : numpy.ndarray, shape (n, 4) or str
			The nucleotide sequence. Either a one hot encoded matrix of 
			nucleotides with n being the size of the chromosome, or a file 
			name for a fasta file.

		dnase : numpy.ndarray, shape (n, 8) or str
			The DNaseI fold change sensitivity. Either an encoded matrix in 
			the manner described in the manuscript or the file name of a 
			bedgraph file.

		regions : numpy.ndarray or None, optional
			The regions of interest to look at. All other regions will be
			ignored. If set to none, the regions of interest are defined
			to be 1kb bins for which all nucleotides are mappable, i.e.
			where there are no n or N symbols in the fasta file. Default
			is None.

		ctxs: list, optional
			The contexts of the gpus to use for prediction. Currently
			prediction is only supported on gpus and not cpus due to
			the time it would take for prediction. For example, if you
			wanted to use three gpus of index 0 1 and 3 (because 2
			is busy doing something else) you would just pass in
			ctxs=[0, 1, 3] and the prediction task will be naturally
			parallelized across your 3 gpus with a linear speedup.

		Returns
		-------
		y : numpy.ndarray, shape=(m, m)
			A matrix of predictions of shape (m, m) where m is the number of
			1kb loci in the chromosome. The predictions will reside in the 
			upper triangle of the matrix since predictions are symmetric.
		"""

		if isinstance(sequence, str) and isinstance(dnase, str):
			if self.verbose:
				print "Converting FASTA"

			sequence = fasta_to_dense(sequence, self.verbose)

			if self.verbose:
				print "Converting DNase"

			dnase = bedgraph_to_dense(dnase, self.verbose)

			if self.verbose:
				print "Encoding DNase"

			dnase = encode_dnase(dnase, self.verbose)

		if regions is None:
			regions = extract_regions(sequence)

		os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
		from .models import predict_task

		Parallel(n_jobs=len(ctxs))( delayed(predict_task)(self.name, 
			self.iteration, ctx, len(ctxs), sequence, dnase, regions, 
			self.use_seq, self.use_dnase, self.use_dist, self.min_dist, 
			self.max_dist, self.batch_size, self.verbose) for ctx in ctxs)

		n = int(regions.max()) / 1000 + 1
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

		import mxnet as mx
		from .models import RambutanSymbol

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

		X_train = mx.io.PrefetchingIter(X_train)

		if validation:
			X_validation = ValidationGenerator(validation_sequence, 
				validation_dnase, validation_contacts, validation_regions,
				batch_size=self.batch_size, min_dist=self.min_dist, 
				max_dist=self.max_dist, use_seq=self.use_seq, 
				use_dnase=self.use_dnase, use_dist=self.use_dist)

		model.fit( X=X_train,
				   eval_data=X_validation if validation else None,
				   eval_metric=eval_metric,
				   batch_end_callback=mx.callback.Speedometer(self.batch_size),
				   kvstore='device',
				   epoch_end_callback=mx.callback.do_checkpoint(self.name)
		)

		self.iteration = self.num_epoch
		return model
