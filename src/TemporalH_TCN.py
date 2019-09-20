import numpy as np
import os

from TemporalH import TemporalH

from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.layers import concatenate, Reshape, Permute
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tcn import TCN



class TemporalH_TCN( TemporalH ):
    def __init__( self, **kwargs ):
        super( TemporalH_TCN , self ).__init__( streams = ['temporal'],
                                                **kwargs )


   
    def _defineNetwork( self ):
        num_feats = int( self.cnnModel.output.shape[1] )
        inp = Input( shape = (self._timesteps, num_feats) )
        y = TCN( nb_filters = 128,
                 return_sequences = False,
                 nb_stacks = 1,
                 dilations = [ 1, 2 ],
                 dropout_rate = 0.3 )( merge )
        y = Dense( self._classes, activation='softmax' )( y )
        
        model = Model( inp, y )
        optimizer = SGD( lr = 1e-2,
                         momentum = 0.9,
                         nesterov = True,
                         decay = 1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model



if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalTCN( flowDataDir  = '/lustre/cranieri/datasets/UCF-101_flow',
                           modelDir     = '/lustre/cranieri/models/ucf101',
                           modelName    = 'model-ucf101-tcn-inception',
                           cnnModelName = 'model-ucf101-optflow-inception',
                           trainListPath = '../splits/ucf101/trainlist01.txt',
                           testListPath  = '../splits/ucf101/testlist01.txt',
                           flowSteps    = 15,
                           clipTh       = 20,
                           restoreModel = False,
                           normalize    = False)

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5,
    #                  storeTests   = True )

    network.train( steps      = 200000,
                   batchSize  = 32,
                   numThreads = 3,
                   maxsize    = 8 )  
