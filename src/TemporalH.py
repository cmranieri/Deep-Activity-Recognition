import os
import numpy as np
from NetworkBase import NetworkBase
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Lambda
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model


class TemporalH( NetworkBase ):
    
    def __init__( self, cnnModelName, adjust = False, **kwargs ):
        self.streams  = kwargs[ 'streams' ]
        self.imuSteps = kwargs[ 'imuSteps' ]
        self.adjust   = adjust
        if cnnModelName is not None:
            cnnPath = os.path.join( kwargs['modelDir'], cnnModelName + '.h5' )
            self.cnnModel = self.loadCNN( cnnPath )
        super( TemporalH , self ).__init__( **kwargs )

    
    def _adjustDim( self, y ):
        ratio = 50 / 15
        indices = [ int( i * ratio ) for i in range( 15 ) ]
        return tf.gather( y, indices, axis=1 )


    def imuBlock( self, shape ):
        inp = Input( shape = shape )
        y = Conv1D(256, 1, padding='same', activation='relu')(inp)
        y = Conv1D(512, 3, padding='same', activation='relu')(y)
        y = MaxPooling1D(2)(y)
        if self.adjust:
            y = Lambda( self._adjustDim )(y)
        # [ b, t, f ]
        model = Model( inputs = inp, outputs = y )
        return model


    def _getFlowFeats( self, batch ):
        batch = np.reshape( batch, [ batch.shape[0],
                                     self.dim, self.dim,
                                     2, self.flowSteps ] )
        # [ t, b, d, d, c ]
        batch = list( np.transpose( batch, [ 4, 0, 1, 2, 3 ] ) )
        featsBatch = self.runFlowCNN( batch )
        return featsBatch


    def _prepareBatch( self, batchDict ):
        inputDataList = list()
        if 'temporal' in self.streams:
            # [ t, b, f ]
            flowFeats = self._getFlowFeats( batchDict[ 'temporal' ] )
            # [ b, t, f ]
            flowFeats = np.transpose( flowFeats, [ 1, 0, 2 ] )
            inputDataList.append( flowFeats )
        if 'inertial' in self.streams:
            imuBatch = batchDict[ 'inertial' ]
            inputDataList.append( imuBatch )
        if len( inputDataList ) == 1:
            return inputDataList[0]
        return inputDataList


    def runFlowCNN( self, batch ):
        featsList = list()
        for batchT in batch:
            feats = self.cnnModel.predict_on_batch( batchT )
            featsList.append( feats )
        # [ t, b, f ]
        featsArray = np.array( featsList )
        return featsArray


    def loadCNN( self, cnnPath ):
        base_model = load_model( cnnPath )
        model = Model( inputs  = base_model.input,
                       outputs = base_model.layers[-2].output )
        return model

