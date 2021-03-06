import numpy as np
import os

from TemporalBase import TemporalBase

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate, Reshape, Permute
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers



class Temporal_LSTM( TemporalBase ):
    def __init__( self, **kwargs ):
        super( Temporal_LSTM , self ).__init__( streams = ['temporal'],
                                                **kwargs )


    def defineNetwork( self ):
        num_feats = int( self.cnnModel.output.shape[1] )
        inp = Input( shape = (self.flowSteps, num_feats) )
        y = LSTM( units = 128,
                  return_sequences = False,
                  dropout = 0.5,
                  unroll = False )( inp )
        y = Dense( self.classes,
                   kernel_regularizer = regularizers.l2( 0.01 ),
                   activation='softmax' )( y )
        
        model = Model( inp, y )
        optimizer = SGD( lr = 1e-2, 
                         momentum=0.9,
                         decay=1e-4,
                         clipnorm=1.,
                         clipvalue=0.5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model



if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = Temporal_LSTM( flowDataDir  = '/lustre/cranieri/datasets/UCF-101_flow',
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
