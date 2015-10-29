# model.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This holds the primary model object, which allows you to build
new graphs, load them from prototxts, train them, or use them
to do prediction.
"""

import subprocess

try:
	import caffe
except:
	pass

from .layers import *
from .solvers import *
from .database import *

class Param( object ):
	"""A parameter, for ease of use."""

	name = "param"

	def __init__( self, **kwargs ):
		self.kwargs = kwargs

	def to_prototxt( self ):
		"""Convert the output to the string format for the prototxt file."""

		prototxt = "  param {\n"
		for key, value in self.kwargs.items():
			prototxt += "    {}: {}\n".format( key, value )
		prototxt += "  }\n"

		return prototxt

class Output( object ):
	"""The output of a network."""

	name = "Output"

	def __init__( self, type, name, bottom, phase=None ):
		self.type = type
		self.name = name
		self.bottom = bottom
		self.phase = phase

	def to_prototxt( self ):
		"""Convert the output to the string format for the prototxt file."""

		prototxt = 'layer {\n' + \
		           '  name: "{}"\n'.format( self.name ) + \
		           '  type: "{}"\n'.format( self.type ) + \
		           '  top: "{}"\n'.format( self.name )

		if isinstance( self.bottom, list ):
			for bottom in self.bottom:
				prototxt += '  bottom: "{}"\n'.format( bottom )
		else:
			prototxt += '  bottom: "{}"\n'.format( bottom )

		if self.phase is not None:
			prototxt += '  include {\n' + \
			            '    phase: {}\n'.format( self.phase ) + \
			            '  }\n'
		prototxt += '}\n'

		return prototxt

class Layer( object ):
	"""A single node, containing a layer, a name, and a pointer to
	nodes above and below."""

	name = "Layer"

	def __init__( self, layer, name, bottom=None, top=None, param=None, phase=None ):
		self.layer = layer 
		self.name = name
		self.bottom = bottom
		self.top = top
		self.param = param
		self.phase = phase

	def to_prototxt( self ):
		"""Convert the node to the string format for the prototxt file."""

		prototxt = 'layer {\n' + \
		           '  name: "{}"\n'.format( self.name ) + \
		           '  type: "{}"\n'.format( self.layer.type ) + \
		           '  top: "{}"\n'.format( self.name )

		if self.top is not None:
			prototxt += '  top: "{}"\n'.format( self.top )

		if isinstance( self.bottom, str ):
			prototxt += '  bottom: "{}"\n'.format( self.bottom )
		elif isinstance( self.bottom, list ):
			for node in self.bottom:
				prototxt += '  bottom: "{}"\n'.format( node )

		if self.phase is not None:
			prototxt += '  include {\n' + \
			            '    phase: {}\n'.format( self.phase ) + \
			            '  }\n'

		if isinstance( self.param, dict ):
			prototxt += '  param {\n'
			for key, val in param.items():
				prototxt += '    {}: {}\n'.format( key, val )
			prototxt += '  }\n'
		elif isinstance( self.param, Param ):
			prototxt += self.param.to_prototxt()
		elif isinstance( self.param, list ):
			for param in self.param:
				if isinstance( param, Param ):
					prototxt += param.to_prototxt()
				elif isinstance( param, dict ):
					prototxt += '  param {\n'
					for key, val in param.items():
						prototxt += '    {}: {}\n'.format( key, val )
					prototxt += '  }\n'					

		prototxt += '  {}\n'.format( self.layer.to_prototxt() )
		prototxt += '}\n'

		if hasattr( self.layer, 'activation' ) and self.layer.activation != 'linear':
			prototxt += '\nlayer {\n' + \
			            '  name: "{}"\n'.format( self.name + '_relu' ) + \
			            '  type: "{}"\n'.format( self.layer.activation ) + \
			            '  bottom: "{}"\n'.format( self.name ) + \
			            '  top: "{}"\n'.format( self.name ) + \
			            '}\n'

		return prototxt


class Model( object ):
	"""Wrapper for the caffe model, exposes a scikit-learn API."""

	name = 'Model'

	def __init__( self, name='caffe-model', policy_name=None ):
		self.name = name
		self.policy_name = policy_name or name + '_policy'
		self.ordered_nodes = []
		self.input_names = []
		self.nodes = {}

	def add_node( self, layer, name, input, params=None, concat_dim=1 ):
		"""Add a node to the graph that is this model."""

		if isinstance( input, list ) and not isinstance( layer, Concat ):
			node = Layer( Concat( concat_dim ), name="pre_{}_concat".format( name ), bottom=input )
			self.nodes["pre_{}_concat".format( name )] = node
			self.ordered_nodes.append( node )

			input = "pre_{}_concat".format( name )

		node = Layer( layer, name, bottom=input )
		self.nodes[name] = node
		self.ordered_nodes.append( node )

	def add_input( self, layer, name, label=None, phase=None ):
		"""Add a node which represents an input to the model."""

		node = Layer( layer, name, top=label, phase=phase )
		self.input_names.append( name )
		self.nodes[name] = node
		self.ordered_nodes.append( node )

	def add_output( self, type, name, input, label='label', phase=None ):
		"""Add a node which represents an output of the model."""

		if isinstance( input, list ):
			input += [label]
		else:
			input = [input, label]

		node = Output( type, name, bottom=input, phase=phase )
		self.nodes[name] = node
		self.ordered_nodes.append( node )

	def compile( self, solver, **kwargs ):
		"""Compile the graph, added all of the 'top' linkers."""

		if isinstance( solver, str ):
			solver = solvers[solver]

		with open( self.name + '.prototxt', 'w' ) as model:
			model.write( self.to_prototxt() )

		with open( self.policy_name + '.prototxt', 'w' ) as policy:
			policy.write( 'net: "{}.prototxt"\n'.format( self.name ) )
			policy.write( solver.to_prototxt() )
			for key, val in kwargs.iteritems():
				policy.write( "{}: {}\n".format( key, val ) )

	def to_prototxt( self ):
		"""Convert the model to a prototxt file."""

		return '\n'.join( node.to_prototxt() for node in self.ordered_nodes )

	def fit( self, X=None, y=None, gpu=None, iterations=None, snapshot=None, weights=None, suffix='' ):
		"""Fit the network to the data.

		Parameters
		----------

		X: <None, str, numpy.ndarray, or dict> The input data. If none, use the parameters specified
			in the prototxt file. If string, replace the input layers source with this string. If
			numpy.ndarray, push the data to a lmdb file (must also specify y as a numpy.ndarray),
			then use that as input. Use a dictionary only if there are multiple input files, and
			then pass in layer name: str/numpy.ndarray pairs, which will be handled in the same
			manner as above. If X is a numpy.ndarray, it must have four dimensions, specified by
			(n_samples, n_channels, height, width).

		y: <None, numpy.ndarray> The targets for each point. Only specify if X is a numpy.ndarray
			or a dictionary of numpy.ndarrays. X.shape[0] must equal y.shape[0]

		gpu: <None, int, str> The gpu to use. If an integer, use that id, if a str, split on
		     commas (for multi-gpu support)

		iterations: <None, int> The number of iterations to train the model for. If none, default
		            to what is used in the prototxt file.

		snapshot: <None, str> If you want to finetune some training, then pass a string to the
		          model file.

		weights: <None, str> If you have prior weights, pass a string to the model file

		suffix: <str> If there are any other commands you want to include in the command
		        line invocation, such as piping output somewhere, include them here.

		Examples
		--------
		model.fit() # Use parameters specified in the policy prototxt file
		model.fit( 'training_data_lmdb', gpu=2, suffix=' | caffe.log' )

		X = numpy.random.randn( 1000, 1, 64, 64 ) # Generate 1000 random 64x64 matrices
		y = numpy.ones(1000)
		y[:500] = 0 # Random labels

		model.fit( X, y )
		"""

		# Make sure we're getting properly formatted data.
		if y is not None and X is None:
			raise ValueError("If y is specified, X must be specified also.")
		if X is not None and not type(X) in (str, dict, numpy.ndarray):
			raise ValueError("X must be a string to a database, a numpy array, "
				             " or a dictionary of layer name: string/numpy.ndarray pairs")
		if isinstance( X, numpy.ndarray ) and not isinstance( y, numpy.ndarray ):
			raise ValueError("If X is a numpy array, y must be passed in as a numpy array "
				             "of targets.")
		if gpu is not None and type(gpu) not in (int, str):
			raise ValueError("gpu must be either the integer id, or a string of comma "
				             "separated numbers (e.g. '2,3')")

		# Initial command
		command = 'caffe train -solver={}.prototxt '.format( self.policy_name )

		# If we're being passed new data, we have to modify the prototxt file at
		# the very minimum, and possibly make a new database.
		if X is not None:
			if self.nodes == {}:
				raise ValueError( "Cannot define a new input if reading from "
					              "existing prototxt file." )

			# If we only have one input database change only that layer
			if isinstance( X, str ):
				node = self.nodes[self.input_names[0]]
				node.layer.source = X

			# If we have a numpy array, push that to a lmdb database.
			elif isinstance( X, numpy.ndarray ):
				numpy_to_lmdb( X, y, "caffe_{}_X_lmdb".format( self.name ) )
				node = self.nodes[self.input_names[0]]
				node.layer.source = "caffe_{}_X_lmdb".format( self.name )

			# If we have multiple input databases, change each layer.
			elif isinstance( X, dict ):
				for name, data in X.items():

					# If we're passed a string, just update that path
					if isinstance( data, str ):
						node = self.nodes[ name ]
						node.layer.source = path

					# If we're passed a numpy array, push it to a database and update the path
					elif isinstance( data, numpy.ndarray ):
						numpy_to_lmdb( data, y, "caffe_{}_{}_lmdb".format( self.name, name ) )
						node = self.nodes[ name ]
						node.layer.source = "caffe_{}_{}_lmdb".format( self.name, name )

			# Write the updated model out
			with open( self.name + '.prototxt', 'w' ) as model:
				model.write( self.to_prototxt() )


		if gpu is not None:
			command += '-gpu {} '.format( str(gpu) )
		if iterations is not None:
			command += '-iterations {} '.format( str(iterations) )
		if snapshot is not None:
			command += '-snapshot {} '.format( snapshot )
		if weights is not None:
			command += '-weights {} '.format( weights )
		command += suffix

		subprocess.call( command.split() )

	def predict( self, X ):
		"""Fit to some data by taking in two numpy arrays and getting a dictionary.

		Parameters
		----------
		X: <numpy.ndarray> A 4D array of data, formatted as (n_samples, n_channels,
			height, width).

		Return
		------
		y: <dict> A dictionary of layer name: layer values for each sample, of the
		format (n_samples, n_channels, height, width).
		"""

		model = caffe.Net( self.name, self.policy_name )

		if isinstance(X, numpy.ndarray):
			in_layer = self.nodes[self.input_names[0]]

			model.blobs[ in_layer ].reshape( X.shape )
			model.blobs[ in_layer ].data[...] = X

		elif isinstance(X, dict):
			for name, data in X.items():
				in_layer = self.nodes[name]

				model.blobs[ in_layer ].reshape( data.shape )
				model.blobs[ in_layer ].data[...] = data

		out = model.forward()
		return out 
		
	@classmethod
	def from_prototxt( cls, model, policy=None ):
		"""Link a model object to existing prototxt files."""

		name = model.replace('.prototxt', '')
		policy_name = policy.replace('.prototxt', '') if policy is not None else None
		return cls( name, policy_name=policy_name )
