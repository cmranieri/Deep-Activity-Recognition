import numpy as np
import os

from LateMultimodalBase import LateMultimodalBase

from tensorflow.keras.layers import Input, Dense, LSTM, Activation
from tensorflow.keras.layers import concatenate, Flatten, Reshape, Permute, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


class LateMultimodal( LateMultimodalBase ):
    def __init__( self, imuShape, homeShape, **kwargs ):
        self.imuShape  = imuShape
        self.homeShape = homeShape
        super( LateMultimodal , self ).__init__( streams = ['temporal', 'inertial', 'smart_home'],
                                                 **kwargs )


    def defineNetwork( self ):
        flowModel = self.flowBlock()
        imuModel  = self.imuBlock( self.imuShape )
        y0 = LSTM( units            = 128,
                  return_sequences  = False,
                  dropout           = 0.7,
                  )( flowModel.output )
        y1 = LSTM( units            = 128,
                  return_sequences  = False,
                  dropout           = 0.7,
                  )( imuModel.output )
        
        # Smart home block
        #homeInp = Input( self.homeShape )
        #y2 = Flatten()( homeInp )
        #y2 = Dense( 128, activation='relu' )( y2 )
        #y2 = Dropout(0.5)( y2 )

        # Late fusion
        y = concatenate( [ y0, y1 ] )
        y = Dense( self.classes,
                   kernel_regularizer = regularizers.l2( 0.01 ),
                   activation = 'softmax' )( y )
       
        model = Model( [flowModel.inputs, imuModel.inputs], y )
        optimizer = SGD( lr        = 1e-2, 
                         momentum  = 0.9,
                         decay     = 1e-4,
                         clipnorm  = 1.,
                         clipvalue = 0.5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
