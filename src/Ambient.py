import numpy as np
import os

from TemporalBase import TemporalBase

from tensorflow.keras.layers import Input, Dense, LSTM, Activation
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import concatenate, Flatten, Reshape, Permute, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


class Ambient( TemporalBase ):
    def __init__( self, imuShape, homeShape, **kwargs ):
        self.imuShape  = imuShape
        self.homeShape = homeShape
        kwargs[ 'cnnModelName' ] = None
        super( Ambient , self ).__init__( streams = ['smart_home'],
                                                **kwargs )


    def defineNetwork( self ):
        homeInp = Input( self.homeShape )
        y2 = Flatten()( homeInp )
        y2 = Dense( 512, activation='relu' )( y2 )
        y2 = Dropout(0.5)( y2 )
        y2 = Dense( 256, activation='relu' )( y2 )
        y2 = Dropout(0.5)( y2 )
        
        y2 = Dense( self.classes,
                    kernel_regularizer = regularizers.l2( 0.01 ),
                    activation = 'softmax' )( y2 )
       
        model = Model( homeInp, y2 )
        optimizer = SGD( lr        = 1e-2, 
                         momentum  = 0.9,
                         decay     = 1e-4,
                         clipnorm  = 1.,
                         clipvalue = 0.5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
