# model.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This holds the primary model object, which allows you to build
new graphs, load them from prototxts, train them, or use them
to do prediction.
"""

import subprocess
#import caffe

from .layers import *
from .solvers import *

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

	def compile( self, solver, **kwargs ):
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

	def fit( self, source=None, gpu=None, iterations=None, snapshot=None, weights=None, suffix='' ):
		"""Fit the network to the data."""

		command = 'caffe train solver={}'.format( self.policy_name )

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


		if gpu is not None:
			command += '-gpu {}'.format( str(gpu) )
		if iterations is not None:
			command += '-iterations {}'.format( str(iterations) )
		if snapshot is not None:
			command += '-snapshot {}'.format( snapshot )
		if weights is not None:
			command += '-weights {}'.format( weights )
		command += suffix

		subprocess.call( command.split() )
		

	@classmethod
	def from_prototxts( cls, model, policy=None ):
		"""Link a model object to existing prototxt files."""

		name = model.replace('.prototxt', '')
		policy_name = policy.replace('.prototxt', '') if policy is not None else None
		return cls( name, policy_name=policy_name )
