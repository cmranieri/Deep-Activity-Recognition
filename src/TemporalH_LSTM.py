import numpy as np
import os

from TemporalH import TemporalH

from keras.layers import Input, Dense, LSTM
from keras.layers import concatenate, Reshape, Permute
from keras.optimizers import SGD
from keras.models import Model



class TemporalH_LSTM( TemporalH ):
    def __init__( self, **kwargs ):
        super( TemporalH_LSTM , self ).__init__( **kwargs )


    def _defineNetwork( self ):
        num_feats = int( self.cnnModel.output.shape[1] )
        inp = Input( shape = (self._timesteps, num_feats) )
        y = LSTM( units = 128,
                  return_sequences = False,
                  dropout = 0.3,
                  unroll = False )( inp )
        y = Dense( self._classes, activation='softmax' )( y )
        
        model = Model( inp, y )
        optimizer = SGD( lr = 1e-2,
                         momentum = 0.9,
                         nesterov = False,
                         decay = 1e-7 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model



if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalH_LSTM( flowDataDir  = '/lustre/cranieri/datasets/UCF-101_flow',
                              modelDir     = '/lustre/cranieri/models/ucf101',
                              modelName    = 'model-ucf101-hlstm-inception',
                              cnnModelName = 'model-ucf101-optflow-inception',
                              timesteps    = 8,
                              restoreModel = False,
                              normalize    = False )

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5 )
    network.train( steps      = 200000,
                   batchSize  = 16,
                   numThreads = 12,
                   maxsize    = 32 )
