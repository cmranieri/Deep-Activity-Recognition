import tensorflow as tf

class Layers:

    def initializeWeights( self, shape, uniform = False, seed = None ):
        initializer = tf.contrib.layers.xavier_initializer( uniform = uniform,
                                                            seed = seed )
        W = tf.get_variable( name = 'weights',
                             shape = shape,
                             initializer = initializer )
        return W


    def initializeBias( self, shape, value = 0.0 ):
        constant =  tf.constant( value , shape = shape )
        return tf.Variable( constant , name = 'bias' )


    def setDefaultInput( self, input = None, channels = None ):
        self.defaultInput  = input
        self.inputChannels = channels


    def getInput( self, input ):
        if input is None:
            return self.defaultInput
        return input


    def getInputChannels( self, in_channels ):
        if in_channels is None:
            return self.inputChannels
        return in_channels


    def shape( self, x ):
        if isinstance( x, tuple ): return np.shape( x )
        return x.get_shape().as_list()


    def conv2d( self,
                out_channels,
                ksize_conv, stride_conv = 1,
                input = None,
                ksize_pool = 2, stride_pool = 2,
                in_channels = None,
                relu = True, bn_phase = None,
                padding = 'SAME',
                scope = 'conv'):
        
        input = self.getInput( input )
        in_channels = self.getInputChannels( in_channels )
        with tf.variable_scope( scope ):
            conv_shape = [ ksize_conv  , ksize_conv,
                           in_channels , out_channels ]
            W = self.initializeWeights( conv_shape )
            b = self.initializeBias( [ out_channels ] )
            conv = tf.nn.conv2d( input, W,
                                 strides=[ 1, stride_conv, stride_conv, 1 ],
                                 padding=padding,
                                 name = 'conv')
            conv = conv + b
            if bn_phase is not None:
                conv = tf.contrib.layers.batch_norm( conv,
                                                     center = True, scale = True,
                                                     is_training = bn_phase,
                                                     scope = 'bn')
            if relu is not None:
                conv = tf.nn.relu( conv, 'relu' )
            if ksize_pool is not None:
                conv = self.pool2d( conv, ksize = ksize_pool, stride = stride_pool )

            self.setDefaultInput( conv, out_channels )
            return conv


    def lrn ( self,
              out_channels,
              input = None,
              depth_radius = 5,
              bias = 2,
              alpha = 10e-4,
              beta = 0.75,
              scope = 'lrn'):

        input = self.getInput( input )
        with tf.variable_scope( scope ):
            lrn = tf.nn.lrn( input,
                             depth_radius = depth_radius,
                             bias  = bias,
                             alpha = alpha,
                             beta  = beta,
                             name  = 'lrn' )
            self.setDefaultInput( lrn, out_channels )
            return lrn


    def pool2d( self, input,
                ksize, stride,
                padding = 'SAME',
                scope = 'pool' ):

        with tf.variable_scope( scope ):
            pool = tf.nn.max_pool( input,
                                   ksize   = [ 1, ksize , ksize , 1 ],
                                   strides = [ 1, stride, stride, 1 ],
                                   padding = padding,
                                   name = 'pool')
            return pool



    def fully( self, out_channels,
               input = None,
               in_channels = None,
               activation = 'relu', dropout = None,
               bn_phase = None, scope = 'fully' ):

        input = self.getInput( input )
        in_channels = self.getInputChannels( in_channels )
        with tf.variable_scope( scope ):
            fully_shape = [ in_channels, out_channels ]
            W = self.initializeWeights( fully_shape )
            b = self.initializeBias( [ out_channels ] )
            fully = tf.matmul( input, W ) + b
            if bn_phase is not None:
                fully = tf.contrib.layers.batch_norm( fully,
                                                      center = True, scale = True,
                                                      is_training = bn_phase,
                                                      scope = 'bn')
            if   activation == 'relu':    fully = tf.nn.relu( fully )
            elif activation == 'softmax': fully = tf.nn.softmax( fully )

            if dropout is not None: fully = tf.nn.dropout( fully , dropout )
            
            self.setDefaultInput( fully , out_channels )
            return fully



    def recurrent( self,
                   out_channels, num_cells,
                   input = None,
                   in_channels = None,
                   in_dropout = 1.0, out_dropout = 1.0,
                   activation = 'relu',
                   scope = 'recurrent'  ):

        input = self.getInput( input )
        in_channels = self.getInputChannels( in_channels )
        with tf.variable_scope( scope ):
            cells = list()
            for _ in range( num_cells ):
                cells += [ tf.contrib.rnn.BasicLSTMCell(
                                out_channels,
                                forget_bias = 1.0,
                                state_is_tuple = True  ) ]
            cell = tf.contrib.rnn.MultiRNNCell( cells,
                                                state_is_tuple = True )
            cell = tf.contrib.rnn.DropoutWrapper( cell,
                                                  input_keep_prob = in_dropout,
                                                  output_keep_prob = out_dropout )
            outputs , states = tf.nn.dynamic_rnn( cell , input, dtype = tf.float32 )
            
            shape = self.shape( outputs )
            trans = list( range( len( shape ) ) )
            trans[0], trans[1] = trans[1], trans[0]

            recur = tf.transpose( outputs, trans )[-1]
            if activation == 'relu': recur = tf.nn.relu( recur )

            self.setDefaultInput( recur , out_channels )
            return recur
