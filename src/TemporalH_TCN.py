import numpy as np
import os

from TemporalH2 import TemporalH

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate, Reshape, Permute
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tcn import TCN


class TemporalH_TCN( TemporalH ):
    def __init__( self, **kwargs ):
        super( TemporalH_TCN , self ).__init__( streams = ['temporal'],
                                                 **kwargs )


    def defineNetwork( self ):
        num_feats = int( self.cnnModel.output.shape[1] )
        inp = Input( shape = (self.flowSteps, num_feats) )
        y = TCN( nb_filters       = 128,
                 nb_stacks        = 1,
                 kernel_size      = 3,
                 use_skip_connections = False,
                 return_sequences = False,
                 dropout_rate     = 0.5,
                 dilations        = [ 1, 2, 4, 8, 16 ] )( inp )
        y = Dense( self.classes, activation='softmax' )( y )
        
        model = Model( inp, y )
        optimizer = SGD( lr = 1e-2 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model



if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalH_TCN( flowDataDir  = '/lustre/cranieri/datasets/UCF-101_flow',
                              modelDir     = '/lustre/cranieri/models/ucf101',
                              modelName    = 'model-ucf101-hlstm-inception',
                              cnnModelName = 'model-ucf101-optflow-inception',
                              trainListPath = '../splits/ucf101/trainlist01.txt',
                              testListPath  = '../splits/ucf101/testlist01.txt',
                              flowSteps    = 15,
                              clipTh       = 20,
                              restoreModel = False,
                              normalize    = False )
    print( network.model.count_params() )
    #network.evaluate( numSegments  = 5,
    #                  maxsize = 128 )
    network.train( steps      = 200000,
                   batchSize  = 32,
                   numThreads = 12,
                   maxsize    = 32 )
