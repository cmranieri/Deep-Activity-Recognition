import numpy as np
import os

from TemporalBase import TemporalBase

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate, Reshape, Permute
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tcn import TCN


class Inertial_TCN( TemporalBase ):
    def __init__( self, imuShape, **kwargs ):
        self.imuShape = imuShape
        kwargs[ 'cnnModelName' ] = None
        super( Inertial_TCN , self ).__init__( streams = ['inertial'],
                                                **kwargs )


    def defineNetwork( self ):
        imuModel = self.imuBlock( self.imuShape )
        y = TCN( nb_filters           = 128,
                 nb_stacks            = 3,
                 kernel_size          = 3,
                 use_skip_connections = True,
                 return_sequences     = False,
                 dropout_rate         = 0.3,
                 dilations            = [ 1, 2, 4, 8 ] )( imuModel.outputs[0] )
        y = Dense( self.classes,
                   kernel_regularizer = regularizers.l2( 0.01 ),
                   activation='softmax' )( y )
       
        model = Model( imuModel.inputs, y )
        optimizer = SGD( lr        = 1e-2,
                         momentum  = 0.9,
                         decay     = 1e-4,
                         clipnorm  = 1.,
                         clipvalue = 0.5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
