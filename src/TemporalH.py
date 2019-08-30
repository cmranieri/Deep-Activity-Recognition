import os
import numpy as np
from NetworkBase import NetworkBase
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model



class TemporalH( NetworkBase ):
    
    def __init__( self, cnnModelName, **kwargs ):
        cnnPath = os.path.join( kwargs['modelDir'], cnnModelName + '.h5' )
        self.cnnModel = self.loadCNN( cnnPath )
        super( TemporalH , self ).__init__( stream = 'temporal', **kwargs )


    def _prepareBatch( self, batch ):
        batch = np.reshape( batch, [ batch.shape[0],
                                     self._dim, self._dim,
                                     2, self._timesteps ] )
        # [ t, b, d, d, c ]
        batch = list( np.transpose( batch, [ 4, 0, 1, 2, 3 ] ) )
        featsBatch = self.runCNN( batch )
        return featsBatch


    def runCNN( self, batch ):
        featsList = list()
        for batchT in batch:
            feats = self.cnnModel.predict_on_batch( batchT )
            featsList.append( feats )
        featsArray = np.array( featsList )
        # [ b, t, f ]
        featsArray = np.transpose( featsArray, [ 1, 0, 2 ] )
        return featsArray


    def loadCNN( self, cnnPath ):
        base_model = load_model( cnnPath )
        base_model.layers.pop()
        model = Model( inputs  = base_model.input,
                       outputs = base_model.layers[-1].output )
        return model

