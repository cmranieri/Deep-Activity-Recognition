# -*- coding: utf-8 -*-

import tensorflow as tf
import Layers
import os


class Temporal:
    def __init__( self,
                  dim = 224,
                  inp_channels = 2,
                  timesteps = 10,
                  n_actions = 101,
                  modelsPath = '../models/',
                  metaGraphName = 'model.meta',
                  restoreModel  = False,
                  seed = None):
        if seed:
            tf.set_random_seed( seed )
        
        self.dim          = dim
        self.inp_channels = inp_channels
        self.timesteps    = timesteps
        self.n_actions    = n_actions
        self.modelsPath   = modelsPath
        
        self.sess = tf.InteractiveSession()

        if not restoreModel:
            self.buildGraph()
            self.sess.run( tf.global_variables_initializer() )
            tf.train.export_meta_graph( filename = os.path.join( modelsPath,
                                                                 metaGraphName ) )
            self.saver = tf.train.Saver( max_to_keep = 2 )
        else:
            self.saver = tf.train.import_meta_graph( os.path.join( modelsPath,
                                                                   metaGraphName ) )
            self.saver.restore( self.sess,
                                tf.train.latest_checkpoint( modelsPath ) )

        writer = tf.summary.FileWriter( 'tmp/net' )
        writer.add_graph( self.sess.graph )



    def buildGraph( self ):
        layers = Layers.Layers()
        
        # Placeholders for input and output
        self.y = tf.placeholder( tf.float32 ,
                                 shape = [ None,
                                           self.n_actions ],
                                 name = 'y' )
        self.x = tf.placeholder( tf.float32,
                                 shape = [ None,
                                           self.dim * self.dim * self.inp_channels * self.timesteps ],
                                 name = 'x' )

        # Phase placeholder for batch normalization
        phase = tf.placeholder( tf.bool , name = 'phase' )

        # Dropout placeholders
        dropout1    = tf.placeholder( tf.float32 , name = 'dropout1' )
        dropout2    = tf.placeholder( tf.float32 , name = 'dropout2' )

        # Preparing network input
        btc  = tf.shape( self.x )[0]
        fold = tf.reshape( self.x , [ btc , self.dim , self.dim , self.inp_channels * self.timesteps ] )
        
        # Convolution and pooling layers
        layers.setDefaultInput( fold , self.inp_channels * self.timesteps )

        conv1 = layers.conv2d( ksize_conv  = 7  , stride_conv  = 2,
                               ksize_pool  = 2  , out_channels = 96,
                               bn_phase = None  , scope = 'conv1' )
        conv1 = layers.lrn( out_channels = 96, scope = 'conv1' )

        conv2 = layers.conv2d( ksize_conv  = 5  , stride_conv  = 2,
                               ksize_pool  = 2  , out_channels = 256,
                               bn_phase = None  ,  scope = 'conv2' )

        conv3 = layers.conv2d( ksize_conv  = 3   , stride_conv  = 1,
                               ksize_pool = None , out_channels = 512,
                               bn_phase = None  , scope = 'conv3' )

        conv4 = layers.conv2d( ksize_conv  = 3    , stride_conv  = 1,
                               ksize_pool  = None , out_channels = 512,
                               bn_phase = None   , scope = 'conv4' )

        conv5 = layers.conv2d( ksize_conv  = 3  , stride_conv  = 1,
                               ksize_pool  = 2  , out_channels = 512,
                               bn_phase = None , scope = 'conv5' )

        flatten = tf.reshape( conv5 , [ btc , 7 * 7 * 512 ] )
        layers.setDefaultInput( flatten , 7 * 7 * 512 )

        # Fully connected layers
        fully1 = layers.fully( out_channels = 4096 , dropout = dropout1,
                               bn_phase = phase    , scope = 'fully1' )
        fully2 = layers.fully( out_channels = 2048 , dropout = dropout2,
                               bn_phase = phase    , scope = 'fully2' )

        # Readout layer
        self.y_ = layers.fully( out_channels = self.n_actions,
                             activation = 'softmax',
                             scope = 'y_' )
        
        # Number of steps trained so far
        global_step = tf.Variable( 0,
                                   name = 'global_step',
                                   trainable = False )
        # Define operations and related tensors
        self.defineOperations()


    def defineOperations( self ):
        learning_rate = tf.placeholder( tf.float32, name = 'learning_rate' )

        w_fc1 = tf.get_default_graph().get_tensor_by_name( 'fully1/weights:0' )
        w_fc2 = tf.get_default_graph().get_tensor_by_name( 'fully2/weights:0' )
        w_out = tf.get_default_graph().get_tensor_by_name( 'y_/weights:0' )
        l2_loss = 1e-3 * ( tf.nn.l2_loss( w_fc1 ) +
                           tf.nn.l2_loss( w_fc2 ) +
                           tf.nn.l2_loss( w_out ) )
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits( labels = self.y,
                                                     logits = self.y_,
                                                     name   = 'cross_entropy') + l2_loss,
                                   name = 'loss' )

        # Train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies( update_ops ):
            optimizer = tf.train.MomentumOptimizer( learning_rate = learning_rate,
                                                    momentum = 0.9,
                                                    name = 'optimizer' )
            self.train_step = optimizer.minimize( loss,
                                                  global_step = self.getGlobalStep(),
                                                  name = 'train_step' )
        
        # Checks whether prediction is correct
        correct_prediction = tf.equal( tf.argmax( self.y_ , 1 ),
                                       tf.argmax( self.y  , 1 ),
                                       name = 'correct_prediction' )

        # Calculates accuracy
        accuracy = tf.reduce_mean( tf.cast( correct_prediction , tf.float32 ),
                                   name = 'accuracy')

        # Builds confusion matrix
        confusion = tf.confusion_matrix( labels      = tf.argmax( self.y  , 1 ) ,
                                         predictions = tf.argmax( self.y_ , 1 ) ,
                                         num_classes = self.n_actions )



    def getGlobalStep( self ):
        return tf.get_default_graph().get_tensor_by_name( 'global_step:0' )


    def saveModel( self ):
        self.saver.save( self.sess,
                         self.modelsPath,
                         write_meta_graph = False,
                         global_step = self.getGlobalStep().eval() )



    def trainBatch( self, x, y,
                    dropout1=0.5, dropout2=0.5,
                    in_drop=1.0, out_drop=1.0,
                    learning_rate = 1e-2):
        graph      = tf.get_default_graph()
        train_step = graph.get_operation_by_name( 'train_step' )
        accuracy   = graph.get_tensor_by_name(    'accuracy:0' )
        loss       = graph.get_tensor_by_name(    'loss:0' )

        return self.sess.run( [ train_step, accuracy, loss ],
                                feed_dict = { 'x:0': x ,
                                              'y:0': y,
                                              'phase:0': 1,
                                              'dropout1:0':    dropout1,
                                              'dropout2:0':    dropout2,
                                              'learning_rate:0': learning_rate } )


    def evaluateBatch( self, x, y ):
        graph = tf.get_default_graph()
        accuracy = graph.get_tensor_by_name( 'accuracy:0' )
        loss     = graph.get_tensor_by_name( 'loss:0' )

        return self.sess.run( [ accuracy, loss ], 
                                feed_dict = { 'x:0': x, 
                                              'y:0': y,
                                              'phase:0': 0,
                                              'dropout1:0':    1.0,
                                              'dropout2:0':    1.0 } )


    def evaluateActivs( self, x, y ):
        graph = tf.get_default_graph()
        y_ = graph.get_tensor_by_name( 'y_/Softmax:0' )

        return self.sess.run( [ y_ ], 
                                feed_dict = { 'x:0': x, 
                                              'y:0': y,
                                              'phase:0': 0,
                                              'dropout1:0':    1.0,
                                              'dropout2:0':    1.0 } )


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    network = Temporal()
