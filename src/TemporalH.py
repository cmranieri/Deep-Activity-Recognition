import os
import numpy as np
from NetworkBase import NetworkBase
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model



class TemporalH( NetworkBase ):
    
    def __init__( self, cnnModelName, **kwargs ):
        cnnPath = os.path.join( kwargs['modelDir'], cnnModelName + '.h5' )
        self.cnnModel = self.loadCNN( cnnPath )
        self.streams = kwargs[ 'streams' ]
        super( TemporalH , self ).__init__( **kwargs )


    def _getFlowFeats( self, batch ):
        batch = np.reshape( batch, [ batch.shape[0],
                                     self.dim, self.dim,
                                     2, self.flowSteps ] )
        # [ t, b, d, d, c ]
        batch = list( np.transpose( batch, [ 4, 0, 1, 2, 3 ] ) )
        featsBatch = self.runCNN( batch )
        return featsBatch


    def _prepareBatch( self, batchDict ):
        concatList = list()
        if 'temporal' in self.streams:
            # [ t, b, f ]
            videoFeats = self._getFlowFeats( batchDict[ 'temporal' ] )
            # [ f, t, b ]
            videoFeats = np.transpose( videoFeats, [2, 0, 1 ] )
            concatList += list( videoFeats )
        if 'inertial' in self.streams:
            # [ b, t, f ]
            inertialFeats = batchDict[ 'inertial' ]
            # [ f, t, b ]
            inertialFeats = np.transpose( inertialFeats, [2, 1, 0] )
            concatList += list( inertialFeats )
        # [ b, t, f ]
        featsBatch = np.transpose( concatList, [2, 1, 0] )
        return featsBatch


    def runCNN( self, batch ):
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

