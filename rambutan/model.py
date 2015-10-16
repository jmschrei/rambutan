# model.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This holds the primary model object, which allows you to build
new graphs, load them from prototxts, train them, or use them
to do prediction.
"""

import os
#import caffe

class Layer( object ):
	"""A single node, containing a layer, a name, and a pointer to
	nodes above and below."""

	name = "Layer"

	def __init__( self, layer, name, bottom=None, top=None ):
		self.layer = layer 
		self.name = name
		self.bottom = bottom
		self.top = top

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

		prototxt += '  {}\n'.format( self.layer.to_prototxt() )
		prototxt += '}\n'
		return prototxt


class Model( object ):
	"""Wrapper for the caffe model, exposes a scikit-learn API."""

	name = 'Model'

	def __init__( self, name='caffe-model', **kwargs ):

		self.name = name
		self.ordered_nodes = []
		self.nodes = {}
		self.policy = { key: value for key, value in kwargs.items() }


	def add_node( self, layer, name, input=None ):
		"""Add a node to the graph that is this model."""

		node = Layer( layer, name, bottom=input )
		self.nodes[name] = node
		self.ordered_nodes.append( node )

	def compile( self ):
		"""Compile the graph, added all of the 'top' linkers."""

		for name, node in self.nodes.items():
			if node.bottom is not None:
				bottom = self.nodes[node.bottom]

				if bottom.top is None:
					bottom.top = name
				elif isinstance( bottom.top, str ):
					bottom.top = [bottom.top, name]
				elif isinstance( bottom.top, list ):
					bottom.top.append( name )
				else:
					raise ValueError( "bottom.top must be a string or a list or None"
						              "but is a {}".format( type(bottom.top) ) )

	def to_prototxt( self ):
		"""Convert the model to a prototxt file."""

		return '\n'.join( node.to_prototxt() for node in self.ordered_nodes )

	def to_policy_prototxt( self ):
		"""Convert the parameters to a policy prototxt file."""

		return 'net: {}\n'.format( self.name + '.prototxt') + \
		      '\n'.join( '{}: {}'.format( key, value ) for key, value in self.policy.items() )

	def fit( self, source=None, policy=None, gpu=0 ):
		"""Fit the network to the data."""

		if policy is None:
			policy = '{}_policy.prototxt'.format( self.name )
			with open( policy, 'w') as policy:
				policy.write( self.to_policy_prototxt() )

		os.execute('caffe train -solver={} -gpu {}'.format( policy, str(gpu) ))
