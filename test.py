from rambutan import *
import numpy

model = Model()
model.add_input( LMDB(batch_size=1024), "X", label="label" )
model.add_node( InnerProduct(1000), "x_ip1", input="X" )
model.add_output( "Accuracy", "accuracy", input="x_ip1" )
model.compile( solver="ADAM", max_iter=10, display=5 )

X = numpy.random.randn(1024, 1, 64, 64)
y = numpy.ones(1024)
y[:500] = 0

model.fit( X, y )
print model.predict( X )



'''
model.add_input( LMDB('../data/x1_seq_21_lmdb', batch_size=1024), "x1_seq", label="label" )
model.add_input( LMDB('../data/x2_seq_21_lmdb', batch_size=1024), "x2_seq" )
model.add_input( LMDB('../data/x1_dnase_21_lmdb', batch_size=1024), "x1_dnase" )
model.add_input( LMDB('../data/x2_dnase_21_lmdb', batch_size=1024), "x2_dnase" )

model.add_node( Convolution(16, 7, 4), "x1_seq_conv1", input="x1_seq" )
model.add_node( Convolution(16, 7, 4), "x2_seq_conv1", input="x2_seq" )
model.add_node( Convolution(16, 7, 1), "x1_dnase_conv1", input="x1_dnase" )
model.add_node( Convolution(16, 7, 1), "x2_dnase_conv1", input="x2_dnase" )

model.add_node( InnerProduct(1000, activation='ReLU'), "x1_ip1", input=["x1_seq_conv1", "x1_dnase_conv1"] )
model.add_node( InnerProduct(1000, activation='ReLU'), "x2_ip1", input=["x2_seq_conv1", "x2_dnase_conv1"] )

model.add_node( InnerProduct(1000, activation='ReLU'), "x_ip1", input=["x1_ip1", "x2_ip1"] )
model.add_node( InnerProduct(2, activation="softmax"), "y_pred", input="x_ip1" )

model.add_output( "SoftmaxWithLoss", "loss", input="y_pred" )
model.add_output( "Accuracy", "accuracy", input="y_pred" )

model.compile( solver='ADAM', max_iter=1000, display=5 )

#model.fit( { 'x1_seq' : '../data/x1_seq_1_lmdb',
#	         'x2_seq' : '../data/x2_seq_1_lmdb',
#	         'x1_dnase' : '../data/x1_dnase_1_lmdb',
#	         'x2_dnase' : '../data/x2_dnase_1_lmdb' }, gpu=2 )
model.fit( gpu=2 )
'''