.. _rambutan:

Rambutan
========

This file contains the Rambutan model. It is composed of two functions, `fit` and `predict`. 

The `predict` method will use a pre-trained version of Rambutan and predict all pairs of regions in a given sequence file. It takes in nucleotide sequence and DNaseI sensitivity either as bit encoded numpy arrays, or as FastA and bedgraph files respectively. Note that the DNaseI sensitivity should be the fold change signal, not the log10 p-values or raw values. If FastA or bedgraph files are passed in, they are converted internally to the bit encoding. Here is an example of it's usage:

.. code-block:: python

	from rambutan import Rambutan

	model = Rambutan('rambutan', 25, verbose=True)
	y_pred = model.predict('chr21.fa', 'E003-DNase.chr21.fc.signal.bedgraph', ctxs=[0, 1, 2, 3])


This assumes that the directory that you are running the code has the Rambutan model and parameter files (for the 25th iterator), the FastA file for chr21, and the appropriate bedgraph file. It will also use gpus 0, 1, 2, and 3 to make the predictions. If you have fewer gpus, pass in fewer numbers.
 

The `fit` function will accept training (and optionally validation) data either as FastA/bedgraph/HiC files, or as numpy array/numpy array/HiC files. It will automatically convert the FastA and bedgraph files to the bit encoding internally. However, this main require a great deal of memory, so it is recommended that you preprocess your data to numpy arrays, store them to disk, and then feed them in as memory maps so that the entire genome does not need to be stored in RAM for Rambutan to be trained. 

Here is an example of how to pre-encode your data:

.. code-block:: python

	from rambutan.utils import fasta_to_dense
	from rambutan.utils import bedgraph_to_dense
	from rambutan.utils import encode_dnase

	# Process the sequence files first
	sequence_files = ['chr{}.fa'.format(i) for i in range(1, 23)]
	for i, sequence_file in enumerate(sequence_files):
		sequence = fasta_to_dense(sequence_file, verbose=True)
		numpy.save('chr{}.ohe.npy'.format(i), sequence) 

	# Process the DNaseI files next
	dnase_files = ['E003-DNase.chr{}.fc.signal.bedgraph'.format(i) for i in range(1, 23)]
	for i, dnase_file in enumerate(dnase_files):
		dnase = bedgraph_to_dense(dnase_file, verbose=True)
		dnase_encoded = encode_dnase(dnase, verbose=True)
		numpy.save('chr{}.dnase.npy'.format(i), dnase_encoded)


After running this you will now have the nucleotide sequence and DNaseI inputs bit encoded as used in the Rambutan model. You can now load them up as memory maps, which essentially are pointers to where the data lives on disk. This means that the entire array doesn't get loaded into memory, but only subsections are loaded as needed. This will slow down training time, but can be mitigated if a solid state drive is used. Here is an example of fitting using memory maps:

.. code-block:: python

	train_contacts = 'GM12878_combined.res1000.contacts.txt.gz'
	train_sequence = numpy.array([ numpy.load('chr{}.ohe.npy'.format(i), mmap_mode='r') for i in chromosomes ])
	train_dnases   = numpy.array([ numpy.load('chr{}.dnase.npy'.format(i), mmap_mode='r') for i in chromosomes ])

	model = Rambutan()
	model.fit(train_sequence, train_dnase, train_contacts, ctxs=[0, 1, 2, 3])

This will fit the Rambutan model to the given data using four GPUs. Fewer or more GPUs can be used as long as they are in the same machine. Distributed learning is not yet available for training Rambutan models.

API Reference
=============

.. automodule:: rambutan.rambutan
	:members:
	:inherited-members:
