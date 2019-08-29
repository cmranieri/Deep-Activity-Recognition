import os

from NetworkBase import NetworkBase

from keras.layers import Input, Dense 
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model



class TemporalH( NetworkBase ):
    
    def __init__( self, cnnPath, **kwargs ):
        super( TemporalH , self ).__init__( stream = 'temporal', **kwargs )
        self.cnnModel = self.loadCNN( cnnPath )


    def _prepareBatch( self, batch ):
        batch = np.reshape( batch, [ batch.shape[0],
                                     self._dim, self._dim,
                                     2, self._timesteps ] )
        batch = list( np.transpose( batch, [ 4, 0, 1, 2, 3 ] ) )
        featsBatch = self.runCNN( batch )
        return featsBatch


    def runCNN( self, batch ):
        featsList = list()
        for batchT in batch:
            feats = self.cnnModel.predict_on_batch( batchT )
            featsList.append( feats )
        featsArray = np.array( featsList )
        return featsArray


    def loadCNN( self, cnnPath ):
        model = load_model( cnnPath )
        # remove softmax layer
        model.pop()
        return model

