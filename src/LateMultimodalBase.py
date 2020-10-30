import os
import numpy as np
from NetworkBase import NetworkBase
import tensorflow as tf
#import keras.backend as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Lambda, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3 as BaseModel


class LateMultimodalBase( NetworkBase ):
    
    def __init__( self, **kwargs ):
        self.streams   = kwargs[ 'streams' ]
        self.imuSteps  = kwargs[ 'imuSteps' ]
        super( LateMultimodalBase, self ).__init__( **kwargs )


    def prepareInpCNN( self, inp ):
        x = tf.reshape( inp, [ -1, self.dim, self.dim,
                               self.nFlowMaps, self.flowSteps] )
        # [ b, t, x, y, ch ]
        x = tf.transpose( x, [ 0, 4, 1, 2, 3 ] )
        x = tf.reshape( x, [ -1, 
                             self.dim, self.dim, 
                             self.nFlowMaps ] )
        return x


    def restoreOutCNN( self, y ):
        num_feats = y.output.shape[1]
        # [b, t, f]
        x = tf.reshape( y.output, [ -1, self.flowSteps, num_feats ] )
        return x

   
    def flowBlock( self ):
        inp = Input( shape = ( self.dim, self.dim,
                               self.nFlowMaps * self.flowSteps ) )
        y = Lambda( self.prepareInpCNN )( inp )
        model = BaseModel( input_tensor = y,
                           weights = None,
                           classes = self.classes )
        y = model.layers[-2]
        y = Lambda( self.restoreOutCNN )( y )
        model = Model( inputs = inp, outputs = y )
        return model


    def imuBlock( self, shape ):
        inp = Input( shape = shape )
        y = BatchNormalization()( inp )
        y = Conv1D(128, 11, padding='same', activation='relu')(y)
        y = MaxPooling1D(2)(y)
        y = Conv1D(256, 11, padding='same', activation='relu')(y)
        y = MaxPooling1D(2)(y)
        y = BatchNormalization()( y )
        y = Conv1D(378, 11, padding='same', activation='relu')(y)
        y = MaxPooling1D(2)(y)
        # [ b, t, f ]
        model = Model( inputs = inp, outputs = y )
        return model


    def _prepareBatch( self, batchDict ):
        inputDataList = list()
        if 'temporal' in self.streams:
            flowBatch = batchDict[ 'temporal' ]
            inputDataList.append( flowBatch )
        if 'inertial' in self.streams:
            imuBatch = batchDict[ 'inertial' ]
            inputDataList.append( imuBatch )
        if 'smart_home' in self.streams:
            homeBatch = batchDict[ 'smart_home' ]
            inputDataList.append( homeBatch )
        if len( inputDataList ) == 1:
            return inputDataList[0]
        return inputDataList
