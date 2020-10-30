import numpy as np
import os

from TemporalBase import TemporalBase

from tensorflow.keras.layers import Input, Dense, LSTM, Activation
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import concatenate, Flatten, Reshape, Permute, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


class Inertial_Home( TemporalBase ):
    def __init__( self, imuShape, homeShape, **kwargs ):
        self.imuShape  = imuShape
        self.homeShape = homeShape
        kwargs[ 'cnnModelName' ] = None
        super( Inertial_Home , self ).__init__( streams = ['inertial', 'smart_home'],
                                                **kwargs )


    def defineNetwork( self ):
        imuModel = self.imuBlock( self.imuShape )
        y1 = LSTM( units            = 128,
                  return_sequences  = False,
                  dropout           = 0.7,
                  )( imuModel.outputs[-1] )
        
        homeInp = Input( self.homeShape )
        y2 = Flatten()( homeInp )
        y2 = Dense( 128, activation='relu' )( y2 )
        y2 = Dropout(0.5)( y2 )
        y = concatenate( [ y1, y2 ] )
        
        y = Dense( self.classes,
                   kernel_regularizer = regularizers.l2( 0.01 ),
                   activation = 'softmax' )( y )
       
        model = Model( [imuModel.inputs, homeInp], y )
        optimizer = SGD( lr        = 1e-2, 
                         momentum  = 0.9,
                         decay     = 1e-4,
                         clipnorm  = 1.,
                         clipvalue = 0.5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
