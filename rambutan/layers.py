# layers.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This holds all the layers, their initializations, and their conversions
to strings for the prototxt file.
"""

def dict_to_prototxt( d, indent=1 ):
	"""
	Convert a dictionary to a prototxt element.
	"""

	if d is None:
		return ''
	return '{\n' + '\n'.join( [('  '*indent + '{}: {}').format(key, value) 
		for key, value in d.items()] ) + '\n' + '  '*(indent-1) + '}'

###############################################################################
# DATA INPUT LAYERS
###############################################################################

class Database( object ):
	"""A input layer for loading data from LMDB or LevelDB."""

	type = "Data"

	def __init__( self, source, batch_size, backend ):
		self.source = source
		self.batch_size = batch_size
		self.backend = backend

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		return 'data_param {\n' + \
		       '    source: "{}"\n'.format( self.source ) + \
		       '    batch_size: {}\n'.format( self.batch_size ) + \
		       '    backend: {}\n'.format( self.backend ) + \
		       '  }'

class LMDB( object ):
	"""An input layer for loading data from LMDB specifically."""

	type = "Data"

	def __init__( self, source, batch_size ):
		self.source = source
		self.batch_size = batch_size

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		return 'data_param {\n' + \
		       '    source: "{}"\n'.format( self.source ) + \
		       '    batch_size: {}\n'.format( self.batch_size ) + \
		       '    backend: LMDB\n' + \
		       '  }'

class InMemory( object ):
	"""An input layer for loading data from memory, such as numpy."""

	type = "MEMORY_DATA"

	def __init__( self, batch_size, channels, height, width ):
		self.batch_size = batch_size
		self.channels = channels
		self.height = height
		self.width = width

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		return 'memory_data_param {\n' + \
		       '    batch_size: {}\n'.format( self.batch_size ) + \
		       '    channels: {}\n'.format( self.channels ) + \
		       '    height: {}\n'.format( self.height ) + \
		       '    width: {}\n'.format( self.width ) + \
		       '  }'

class HDF5Data( object ):
	"""An input layer for loading data from a HDF5 database."""

	type = "HDF5_DATA"

	def __init__( self, source, batch_size ):
		self.source = source
		self.batch_size = batch_size

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		return 'hdf5_data_param {\n' + \
		       '    source: "{}"\n'.format( self.source ) + \
		       '    batch_size: {}\n'.format( self.batch_size ) + \
		       '  }'

###############################################################################
# CORE LAYERS
###############################################################################

class InnerProduct( object ):
	"""A dense inner product layer."""

	type = "InnerProduct"

	def __init__( self, num_output, bias_term=True, weight=None, bias=None ):
		self.num_output = num_output
		self.bias_term = bias_term
		self.weight = weight
		self.bias = bias

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		weight = dict_to_prototxt( self.weight, 3 )
		bias = dict_to_prototxt( self.bias, 3 )

		prototxt = 'inner_product_param {\n' + \
		           '     num_output: {}\n'.format( self.num_output ) + \
		           '     bias_term: {}\n'.format( str(self.bias_term).lower() )

		if weight is not '':
			prototxt += '    weight_filler {}'.format(weight)
		if bias is not '':
			prototxt += '    bias_filler {}'.format(bias)

		prototxt += '  }'
		return prototxt

class Convolution( object ):
	"""A convolution layer."""

	type = "Convolution"

	def __init__( self, kernel_h, kernel_w, num_output, stride=1, pad=0, group=1, 
		bias_term=True, weight=None, bias=None ):
		self.num_output = num_output
		self.kernel_h = kernel_h
		self.kernel_w = kernel_w
		self.pad_h = pad if isinstance(pad, int) else pad[0]
		self.pad_w = pad if isinstance(pad, int) else pad[1]
		self.stride = stride
		self.weight = weight
		self.bias_term = bias_term
		self.bias = bias
		self.group = 1

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		weight = dict_to_prototxt( self.weight, 3 )
		bias = dict_to_prototxt( self.bias, 3 )


		prototxt =  'convolution_param {\n' + \
		            '    num_output: {}\n'.format( self.num_output ) + \
		            '    kernel_h: {}\n'.format( self.kernel_h ) + \
		            '    kernel_w: {}\n'.format( self.kernel_w ) + \
		            '    stride: {}\n'.format( self.stride ) + \
		            '    pad_h: {}\n'.format( self.pad_h ) + \
		            '    pad_w: {}\n'.format( self.pad_w ) + \
		            '    group: {}\n'.format( self.group ) + \
		            '    bias_term: {}\n'.format( str(self.bias_term).lower() )

		if weight is not '':
			prototxt += '    weight_filler {}'.format(weight)
		if bias is not '':
			prototxt += '    bias_filler {}'.format(bias)

		prototxt += '  }'
		return prototxt

class Pooling( object ):
	"""
	A pooling layer, storing the "pooling param" section of the layer.
	"""

	type = "Pooling"

	def __init__( self, kernel_h, kernel_w, pool="MAX", kernel_size=3, stride=1, pad=0 ):
		self.pool = pool
		self.kernel_h = kernel_h
		self.kernel_w = kernel_w
		self.pad_h = pad if isinstance(pad, int) else pad[0]
		self.pad_w = pad if isinstance(pad, int) else pad[1]
		self.stride = stride

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		return  'pooling_param {\n' + \
		        '    pool: {}\n'.format( self.pool ) + \
		        '    kernel_h: {}\n'.format( self.kernel_h ) + \
		        '    kernel_w: {}\n'.format( self.kernel_w ) + \
		        '    stride: {}\n'.format( self.stride ) + \
		        '    pad_h: {}\n'.format( self.pad_h ) + \
		        '    pad_w: {}\n'.format( self.pad_w ) + \
		        '  }'

class LRN( object ):
	"""
	A local response normalization.
	"""

	type = "LRN"

	def __init__( self, local_size=5, alpha=1, beta=5, norm_region="ACROSS_CHANNELS" ):
		self.local_size = local_size
		self.alpha = alpha
		self.beta = beta
		self.norm_region = norm_region

	def to_prototxt( self ):
		"""Convert the parameters to the string format for the prototxt file."""

		return 'lrn_param {\n' + \
		       '    local_size: {}\n'.format( self.local_size ) + \
		       '    alpha: {}\n'.format( self.alpha ) + \
		       '    beta: {}\n'.format( self.beta ) + \
		       '    norm_region: {}\n'.format( norm_region ) + \
		       '  }'
