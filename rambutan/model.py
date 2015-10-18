# model.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This holds the primary model object, which allows you to build
new graphs, load them from prototxts, train them, or use them
to do prediction.
"""

import os
#import caffe

from .layers import *

class Output( object ):
	"""The output of a network."""

	name = "Output"

	def __init__( self, type, name, bottom=None ):
		self.type = type
		self.name = name
		self.bottom = bottom 

	def to_prototxt( self ):
		"""Convert the output to the string format for the prototxt file."""

		return 'layer {\n' + \
		       '  name: "{}"\n'.format( self.name ) + \
		       '  type: "{}"\n'.format( self.type ) + \
		       '  bottom: "{}"\n'.format( self.bottom ) + \
		       '  top: "{}"\n'.format( self.name ) + \
		       '}\n'

class Layer( object ):
	"""A single node, containing a layer, a name, and a pointer to
	nodes above and below."""

	name = "Layer"

	def __init__( self, layer, name, bottom=None, top=None, params=None ):
		self.layer = layer 
		self.name = name
		self.bottom = bottom
		self.top = top
		self.params = params

	def to_prototxt( self ):
		"""Convert the node to the string format for the prototxt file."""

		prototxt = 'layer {\n' + \
		           '  name: "{}"\n'.format( self.name ) + \
		           '  type: "{}"\n'.format( self.layer.type )

		if isinstance( self.top, str ):
			prototxt += '  top: "{}"\n'.format( self.top )
		elif isinstance( self.top, list ):
			for node in self.top:
				prototxt += '  top: "{}"\n'.format( node )

		if isinstance( self.bottom, str ):
			prototxt += '  bottom: "{}"\n'.format( self.bottom )
		elif isinstance( self.bottom, list ):
			for node in self.bottom:
				prototxt += '  bottom: "{}"\n'.format( node )

		if self.params is not None:
			for param in self.params:
				prototxt += '  param {\n' + \
				            '    name: {}\n'.format( param ) + \
				            '  }'

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

	def __init__( self, name='caffe-model', policy_name=None, **kwargs ):
		self.name = name
		self.policy_name = policy_name or name + '_policy'
		self.ordered_nodes = []
		self.input_names = []
		self.nodes = {}
		self.policy = { key: value for key, value in kwargs.items() }

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

	def add_input( self, layer, name ):
		"""Add a node which represents an input to the model."""

		node = Layer( layer, name )
		self.input_names.append( name )
		self.nodes[name] = node
		self.ordered_nodes.append( node )

	def add_output( self, type, name, input ):
		"""Add a node which represents an output of the model."""

		node = Output( type, name, bottom=input )
		self.nodes[name] = node
		self.ordered_nodes.append( node )

	def compile( self ):
		"""Compile the graph, added all of the 'top' linkers."""

		for name, node in self.nodes.items():
			if node.bottom is not None:
				if isinstance( node.bottom, str ):
					bottom_node = node.bottom
					bottom = self.nodes[bottom_node]

					if bottom.top is None:
						bottom.top = name
					elif isinstance( bottom.top, str ):
						bottom.top = [bottom.top, name]
					elif isinstance( bottom.top, list ):
						bottom.top.append( name )
					else:
						raise ValueError( "bottom.top must be a string or a list or None"
							              "but is a {}".format( type(bottom.top) ) )

				elif isinstance( node.bottom, list ):
					for bottom_node in node.bottom:
						bottom = self.nodes[bottom_node]

						if bottom.top is None:
							bottom.top = name
						elif isinstance( bottom.top, str ):
							bottom.top = [bottom.top, name]
						elif isinstance( bottom.top, list ):
							bottom.top.append( name )
						else:
							raise ValueError( "bottom.top must be a string or a list or None"
								              "but is a {}".format( type(bottom.top) ) )

		with open( self.name + '.prototxt', 'w' ) as model:
			model.write( self.model_to_prototxt() )

		with open( self.policy_name + '.prototxt', 'w' ) as policy:
			policy.write( self.policy_to_prototxt() )

	def model_to_prototxt( self ):
		"""Convert the model to a prototxt file."""

		return '\n'.join( node.to_prototxt() for node in self.ordered_nodes )

	def policy_to_prototxt( self ):
		"""Convert the parameters to a policy prototxt file."""

		return 'net: {}\n'.format( self.name + '.prototxt') + \
		      '\n'.join( '{}: {}'.format( key, value ) for key, value in self.policy.items() )

	def fit( self, source=None, gpu=None, iterations=None, snapshot=None, weights=None, suffix='' ):
		"""Fit the network to the data."""

		if source is not None:
			if self.nodes == {}:
				raise ValueError( "Cannot define a new input if reading from existing prototxt file." )
			if isinstance( source, str ):
				node = self.nodes[self.input_names[0]]
				node.layer.source = source
			elif isinstance( source, dict ):
				for name, path in source.items():
					node = self.nodes[ name ]
					node.layer.source = path
			else:
				raise ValueError( "Source must be a string or a dictionary of node name: path pairs." )

			with open( self.name + '.prototxt', 'w' ) as model:
				model.write( self.model_to_prototxt() )

		command = 'caffe train -solver={}'.format( self.policy_name )
		if gpu is not None:
			command += '-gpu {}'.format( str(gpu) )
		if iterations is not None:
			command += '-iterations {}'.format( str(iterations) )
		if snapshot is not None:
			command += '-snapshot {}'.format( snapshot )
		if weights is not None:
			command += '-weights {}'.format( weights )
		command += suffix

		os.execute(command)

	@classmethod
	def from_prototxts( cls, model, policy=None ):
		"""Link a model object to existing prototxt files."""

		name = model.replace('.prototxt', '')
		policy_name = policy.replace('.prototxt', '') if policy is not None else None
		return network = cls( name, policy_name=policy_name )
