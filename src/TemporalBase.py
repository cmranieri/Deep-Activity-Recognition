import os
import numpy as np
from NetworkBase import NetworkBase
import tensorflow as tf
#import keras.backend as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Lambda, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model


class TemporalBase( NetworkBase ):
    
    def __init__( self, cnnModelName, adjust = False, adjustRatio = (50,15), **kwargs ):
        self.streams     = kwargs[ 'streams' ]
        self.imuSteps    = kwargs[ 'imuSteps' ]
        self.adjust      = adjust
        self.adjustRatio = adjustRatio
        if cnnModelName is not None:
            cnnPath = os.path.join( kwargs['modelDir'], cnnModelName + '.h5' )
            print(cnnPath)
            self.cnnModel    = self.loadCNN( cnnPath )
        super( TemporalBase , self ).__init__( **kwargs )

    
    def _adjustDim( self, y ):
        before, after = self.adjustRatio
        ratio = before / after
        indices = np.arange( 0, before, ratio )
        indices = np.array( indices, dtype=np.int32 )
        return tf.gather( y, indices, axis=1 )


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
        if self.adjust:
            y = Lambda( self._adjustDim )(y)
        #y = BatchNormalization()(y)
        # [ b, t, f ]
        model = Model( inputs = inp,outputs = y )
        return model


    def _getFlowFeats( self, batch ):
        batch = np.reshape( batch, [ -1, self.dim, self.dim,
                                     self.nFlowMaps, self.flowSteps ] )
        # [ b, t, d, d, c ]
        batch = list( np.transpose( batch, [ 0, 4, 1, 2, 3 ] ) )
        featsBatch = self.runFlowCNN( batch )
        return featsBatch


    def _prepareBatch( self, batchDict ):
        inputDataList = list()
        if 'temporal' in self.streams:
            # [ b, t, f ]
            flowFeats = self._getFlowFeats( batchDict[ 'temporal' ] )
            inputDataList.append( flowFeats )
        if 'inertial' in self.streams:
            imuBatch = batchDict[ 'inertial' ]
            inputDataList.append( imuBatch )
        if 'smart_home' in self.streams:
            homeBatch = batchDict[ 'smart_home' ]
            inputDataList.append( homeBatch )
        if len( inputDataList ) == 1:
            return inputDataList[0]
        return inputDataList


    def runFlowCNN( self, batch ):
        featsList = list()
        batch = np.reshape( batch, [ -1, self.dim, self.dim, self.nFlowMaps ] )
        feats = self.cnnModel.predict_on_batch( batch )
        # [ b, t, f ]
        feats = np.reshape( feats, [ -1, self.flowSteps, feats.shape[1] ] )
        return feats


    def loadCNN( self, cnnPath ):
        base_model = load_model( cnnPath )
        model = Model( inputs  = base_model.input,
                       outputs = base_model.layers[-2].output )
        return model

