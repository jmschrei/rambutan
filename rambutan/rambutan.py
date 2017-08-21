# rambutan.py
# Contact: Jacob Schreiber
#          jmschr@cs.washington.edu

import os, numpy, pandas

try:
	from sklearn.metrics import roc_auc_score
except:
	roc_auc_score = 'acc'

from joblib import Parallel, delayed

from .io import TrainingGenerator, ValidationGenerator
from .utils import bedgraph_to_dense, fasta_to_dense
from .utils import encode_dnase, extract_regions

def extract_sequence(filename, verbose=False):
	"""Extract a nucleotide sequence from a file and encode it.

	This function will read in a FastA formatted DNA file and convert it to be
	a one-hot encoded numpy array for internal use. If a one-hot encoded file
	is passed in, it is simply returned. This function is a convenient wrapper
	for joblib to parallelize the unzipping portion.

	Parameters
	----------
	filename : str or numpy.ndarray
		The name of the fasta file to open or the one-hot encoded sequence.

	verbose: bool, optional
		Whether to report the status while extracting sequence. This does not
		look good when done in parallel, so it is suggested it is set to false
		in that case.

	Returns
	-------
	sequence : numpy.ndarray, shape=(n, 4)
		The one-hot encoded DNA sequence.
	"""

	if isinstance(filename, str):
		if verbose:
			print("Converting {}".format(filename))

		return fasta_to_dense(filename, verbose)
	return filename


def extract_dnase(filename, verbose=False):
	"""Extract a DNaseI file and encode it.

	This function will read in a bedgraph format file and convert it to the
	one-hot encoded numpy array used internally. If a one-hot encoded file is
	passed in, it is simple returned. This function is a convenient wrapper for
	joblib to parallelize the unzipping portion.

	Parameters
	----------
	filename : str or numpy.ndarray
		The name of the bedgraph file to open or the one-hot encoded sequence.

	verbose: bool, optional
		Whether to report the status while extracting sequence. This does not
		look good when done in parallel, so it is suggested it is set to false
		in that case.

	Returns
	-------
	sequence : numpy.ndarray, shape=(n, 8)
		The one-hot encoded DNaseI sequence.
	"""

	if isinstance(filename, str):
		if verbose:
			print("Converting {}".format(filename))

		dnase_dense = bedgraph_to_dense(filename, verbose)

		if verbose:
			print("Encoding {}".format(filename))

		dnase_ohe = encode_dnase(dnase_dense, verbose)
		return dnase_ohe
	return filename


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

	model : mxnet.symbol or None
		An alternate neural network can be passed in if one wishes to train that
		using the same framework instead of the original Rambutan model.

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

	def __init__(self, name='rambutan', iteration=None, model=None, 
		learning_rate=0.01,  num_epoch=25, epoch_size=500, wd=0.0, 
		optimizer='adam', batch_size=1024, min_dist=50000, max_dist=1000000, 
		use_seq=True, use_dnase=True, use_dist=True, verbose=True):

		self.name = name
		self.iteration = iteration
		self.model = model
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
				print("Converting FASTA")

			sequence = fasta_to_dense(sequence, self.verbose)

			if self.verbose:
				print("Converting DNase")

			dnase = bedgraph_to_dense(dnase, self.verbose)

			if self.verbose:
				print("Encoding DNase")

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

	def fit(self, sequence, dnase, contacts, regions=None, validation_contacts=None,
		training_chromosome=None, validation_chromosome=None, ctxs=[0], 
		eval_metric=roc_auc_score, symbol=None, n_jobs=1):
		"""Fit the model to sequence, DNaseI, and Hi-C data.

		You can fit the Rambutan model to new data. One must pass in sequence
		data, DNaseI data, and Hi-C contact maps. The sequence data can come
		either in the form of FastA files or one-hot encoded numpy arrays. The
		DNaseI data can likewise come as either bedgraph files or numpy arrays.
		The Hi-C data must come in the traditional 7 column format. Validation
		data can optionally be passed in to report a validation set error during
		the training process. NOTE: Regardless of if they are used or not, all
		chromosomes should be passed in to the `sequence` and `dnase` parameters.
		The contacts specified in `contacts` will dictate which are used. This is
		to make the internals easier.

		Parameters for training such as the number of epochs and batches are
		set in the initial constructor, following with the sklearn format for
		estimators.

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

		if not isinstance(sequence, list):
			raise ValueError("sequence must be a list of FastA file names or pre-encoded numpy arrays.")
		if not isinstance(dnase, list):
			raise ValueError("DNase must be a list of bedgraph file names or pre-encoded numpy arrays.")

		if isinstance(contacts, str):
			contacts = pandas.read_csv(contacts, sep='\t')

		with Parallel(n_jobs=n_jobs) as parallel:
			sequences = parallel( delayed(extract_sequence)(filename, self.verbose) for filename in sequence )
			dnases = parallel( delayed(extract_dnase)(filename, self.verbose) for filename in dnase )

			if regions is None:
				if self.verbose:
					print("Extracting regions")

				regions = parallel( delayed(extract_regions)(sequence) for sequence in sequences )

		sequences = numpy.array(sequences)
		dnases = numpy.array(dnases)
		regions = numpy.array(regions)

		if isinstance(validation_contacts, str):
			validation_contacts = pandas.read_csv(validation_contacts, sep='\t')
			validation_chromosome = int(validation_contacts.ix[0][0][3:])

		import mxnet as mx
		from .models import RambutanSymbol

		if symbol is None:
			symbol = self.model

		model = symbol(ctx=map(mx.gpu, ctxs),
			                   epoch_size=self.epoch_size,
			                   num_epoch=self.num_epoch,
			                   learning_rate=self.learning_rate,
			                   wd=self.wd,
			                   optimizer=self.optimizer
			    )	

		training_contacts = numpy.empty((contacts.shape[0], 3), dtype='float64')
		training_contacts[:,0] = [int(chrom[3:])-1 for chrom in contacts['chr1']]
		training_contacts[:,1] = contacts['fragmentMid1'].values
		training_contacts[:,2] = contacts['fragmentMid2'].values

		if self.verbose:
			print("Training on {} contacts".format(training_contacts.shape[0]))
 
		X_train = TrainingGenerator(sequences, dnases, training_contacts, regions,
			self.batch_size, min_dist=self.min_dist, max_dist=self.max_dist,
			use_seq=self.use_seq, use_dnase=self.use_dnase, 
			use_dist=self.use_dist)

		if validation_contacts is not None:
			validation_contacts = validation_contacts[['fragmentMid1', 'fragmentMid2']].values

			if self.verbose:
				print("Validating on {} contacts from chromosome {}".format(validation_contacts.shape[0], validation_chromosome))

			X_validation = ValidationGenerator(sequences[validation_chromosome-1],
				dnases[validation_chromosome-1], validation_contacts, regions[validation_chromosome-1],
				batch_size=self.batch_size, min_dist=self.min_dist, 
				max_dist=self.max_dist, use_seq=self.use_seq, 
				use_dnase=self.use_dnase, use_dist=self.use_dist)

		model.fit( X=X_train,
				   eval_data=X_validation if validation_contacts is not None else None,
				   eval_metric=eval_metric,
				   batch_end_callback=mx.callback.Speedometer(self.batch_size),
				   kvstore='device',
				   epoch_end_callback=mx.callback.do_checkpoint(self.name)
		)

		self.iteration = self.num_epoch
		return model
