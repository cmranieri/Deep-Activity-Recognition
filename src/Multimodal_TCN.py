import numpy as np
import os

from TemporalH2 import TemporalH

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate, Reshape, Permute
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tcn import TCN

class Multimodal_TCN( TemporalH ):
    def __init__( self, imuShape, **kwargs ):
        self.imuShape = imuShape
        super( Multimodal_TCN , self ).__init__( streams = ['temporal','inertial'],
                                                 **kwargs )


    def defineNetwork( self ):
        numFeatsFlow = int( self.cnnModel.output.shape[1] )
        # [ b, f, t ]
        flowInp = Input( shape = ( self.flowSteps, numFeatsFlow ) )
        imuModel = self.imuBlock( self.imuShape )
        merge = concatenate( [ flowInp, imuModel.outputs[0] ] )
        # [ b, t, f ]
        y = Permute( (2, 1) )( merge )
        y = TCN( nb_filters       = 128,
                 nb_stacks        = 3,
                 kernel_size      = 3,
                 use_skip_connections = True,
                 return_sequences = False,
                 dropout_rate     = 0.5,
                 dilations        = [ 1, 2, 4, 8, 16 ] )( merge )
        y = Dense( self.classes,
                   kernel_regularizer = regularizers.l2( 0.01 ),
                   activation='softmax' )( y )
        
        model = Model( [ flowInp, imuModel.inputs[0] ], y )
        optimizer = SGD( lr = 1e-2,
                         momentum=0.9,
                         decay=1e-4,
                         clipnorm=1.,
                         clipvalue=0.5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
