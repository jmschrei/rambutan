# solvers.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Misc additions to the code.
"""

class Solver( object ):
	"""
	The solver to use.
	"""

	def __init__( self, solver_type, lr, momentum=None, weight_decay=None, gamma=None, stepsize=None, **kwargs ):
		self.solver_type = solver_type
		self.lr = lr
		self.momentum = momentum
		self.gamma = gamma
		self.stepsize = stepsize
		self.weight_decay = weight_decay
		self.kwargs = kwargs

	def to_prototxt( self ):
		prototxt = 'solver_type: {}\n'.format( self.solver_type ) + \
		           'base_lr: {}\n'.format( self.lr )

		if self.momentum is not None:
			prototxt += 'momentum: {}\n'.format( self.momentum )
		if self.weight_decay is not None:
			prototxt += 'weight_decay: {}\n'.format( self.weight_decay )
		if self.gamma is not None:
			prototxt += 'gamma: {}\n'.format( self.gamma )
		if self.stepsize is not None:
			prototxt += 'stepsize: {}\n'.format( self.stepsize )

		for key, val in self.kwargs.iteritems():
			prototxt += '{}: {}\n'.format( key, val )

		return prototxt

class ADAM( Solver ):
	"""The ADAM solver."""

	def __init__( self, lr, momentum=None, weight_decay=None, gamma=None, stepsize=None ):
		super( ADAM, self ).__init__( solver_type='ADAM',
		                              lr=lr,
		                              momentum=momentum,
		                              weight_decay=weight_decay,
		                              gamma=gamma,
		                              stepsize=stepsize )

class Adagrad( Solver ):
	"""Adaptive gradient descent."""

	def __init__( self, lr, momentum=None, weight_decay=None, gamma=None, stepsize=None ):
		super( Adagrad, self ).__init__( solver_type='ADAGRAD',
		                                 lr=lr,
		                                 momentum=momentum,
		                                 weight_decay=weight_decay,
		                                 gamma=gamma,
		                                 stepsize=stepsize )

class Nesterov( Solver ):
	"""Nesterov momentum."""

	def __init__( self, lr, momentum=None, weight_decay=None, gamma=None, stepsize=None ):
		super( Nesterov, self ).__init__( solver_type='NESTEROV',
		                                  lr=lr,
		                                  momentum=momentum,
		                                  weight_decay=weight_decay,
		                                  gamma=gamma,
		                                  stepsize=stepsize )

class SGD( Solver ):
	"""Stochastic gradient descent."""

	def __init__( self, lr, momentum=None, weight_decay=None, gamma=None, stepsize=None ):
		super( SGD, self ).__init__( solver_type='SGD',
		                             lr=lr,
		                             momentum=momentum,
		                             weight_decay=weight_decay,
		                             gamma=gamma,
		                             stepsize=stepsize )

solvers = { 
            'ADAM': ADAM( lr=0.001 ),
            'Adagrad': Adagrad( lr=0.001 ),
            'Nesterov': Nesterov( lr=0.001 ),
            'SGD': SGD( lr=0.001 )
           }