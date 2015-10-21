from rambutan import *

model = Model()

model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x1_seq_chr21_train_lmdb', batch_size=1024), "x1_seq", phase="TRAIN", label="label" )
model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x2_seq_chr21_train_lmdb', batch_size=1024), "x2_seq", phase="TRAIN" )
model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x1_dnase_chr21_train_lmdb', batch_size=1024), "x1_dnase", phase="TRAIN" )
model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x2_dnase_chr21_train_lmdb', batch_size=1024), "x2_dnase", phase="TRAIN" )

model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x1_seq_chr21_train_lmdb', batch_size=1024), "x1_seq", phase="TEST", label="label" )
model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x2_seq_chr21_train_lmdb', batch_size=1024), "x2_seq", phase="TEST" )
model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x1_dnase_chr21_train_lmdb', batch_size=1024), "x1_dnase", phase="TEST" )
model.add_input( LMDB('/net/noble/vol1/home/jmschr/proj/contact/data/x2_dnase_chr21_train_lmdb', batch_size=1024), "x2_dnase", phase="TEST" )

model.add_node( Convolution(7, 4, num_output=16), "x1_seq_conv1", input="x1_seq" )
model.add_node( Convolution(7, 4, num_output=16), "x2_seq_conv1", input="x2_seq" )
model.add_node( Convolution(7, 1, num_output=16), "x1_dnase_conv1", input="x1_dnase" )
model.add_node( Convolution(7, 1, num_output=16), "x2_dnase_conv1", input="x2_dnase" )

model.add_node( InnerProduct(1000, activation='ReLU'), "x1_ip1", input=["x1_seq_conv1", "x1_dnase_conv1"] )
model.add_node( InnerProduct(1000, activation='ReLU'), "x2_ip1", input=["x2_seq_conv1", "x2_dnase_conv1"] )

model.add_node( InnerProduct(1000, activation='ReLU'), "x_ip1", input=["x1_ip1", "x2_ip1"] )
model.add_node( InnerProduct(2), "y_pred", input="x_ip1" )

model.add_output( "SoftmaxWithLoss", "loss", input="y_pred" )
model.add_output( "Accuracy", "accuracy", input="y_pred" )

model.compile( solver='ADAM', max_iter=10000, test_iter=15, test_interval=140 )
model.fit( gpu=2 )