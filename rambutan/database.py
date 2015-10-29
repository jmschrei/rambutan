# database.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Database wrapper for arrays. Used to convert numpy arrays to
databases for caffe to read.
"""

import numpy

try:
	import lmdb
	import caffe
except:
	pass

def numpy_to_lmdb( X, y, source ):
	"""Quick wrapper to push numpy arrays to a new LMDB database."""

	assert len(X.shape) == 4
	assert X.shape[0] == y.shape[0]

	db = LMDB( source )
	db.push( X, y )
	db.close()

class LMDB( object ):
	"""A LMDB database wrapper."""

	type = "Data"

	def __init__( self, source=None, batch_size=None, map_size=1.e12 ):
		self.source = source
		self.batch_size = batch_size

		self.lmdb = None
		if self.batch_size is None:
			self.lmdb = lmdb.open( source, map_size=map_size )
		
		self.counter = 0

	def push( self, X, y, overwrite=True ):
		"""Push some data to the open LMDB database."""

		n = X.shape[0]
		assert X.shape[0] == y.shape[0]
		assert len(X.shape) == 4

		data_buffer = numpy.empty( n, dtype=tuple )

		for i in xrange(n):
			str_id = "{:12}".format( self.counter )

			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = X.shape[1]
			datum.height = X.shape[2]
			datum.width = X.shape[3]

			datum.data = X[i].tobytes()
			datum.label = int(y[i])

			data_buffer[i] = ( str_id, datum.SerializeToString() )
			self.counter += 1

		with self.lmdb.begin( write=True ) as db:
			cur = db.cursor()
			cur.putmulti( data_buffer, overwrite=overwrite )

	def close( self ):
		"""Close the underlying lmdb object."""

		self.lmdb.close()

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		if self.batch_size is None:
			raise ValueError("Must specify batch_size to convert to prototxt.")

		return 'data_param {\n' + \
		       '    source: "{}"\n'.format( self.source ) + \
		       '    batch_size: {}\n'.format( self.batch_size ) + \
		       '    backend: LMDB\n' + \
		       '  }'
