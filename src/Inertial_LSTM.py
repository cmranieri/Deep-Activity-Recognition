import numpy as np
import os

from TemporalH import TemporalH

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate, Reshape, Permute
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model


class Inertial_LSTM( TemporalH ):
    def __init__( self, imuShape, **kwargs ):
        self.imuShape = imuShape
        super( Inertial_LSTM , self ).__init__( streams = ['inertial'],
                                                cnnModelName = None,
                                                **kwargs )


    def defineNetwork( self ):
        imuModel = self.imuBlock( self.imuShape )
        y = LSTM( units = 128,
                  return_sequences = False,
                  dropout = 0.3,
                  unroll = False )( imuModel.outputs[-1] )
        y = Dense( self.classes, activation='softmax' )( y )
        
        model = Model( imuModel.inputs, y )
        optimizer = SGD( lr = 1e-3,
                         momentum = 0.9,
                         nesterov = True,
                         decay = 1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = 'rmsprop',
                       metrics   = [ 'acc' ] ) 
        return model



if __name__ == '__main__':
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalH_LSTM( flowDataDir  = '/lustre/cranieri/datasets/multimodal_dataset_flow',
                              modelDir     = '/lustre/cranieri/models/multimodal',
                              modelName    = 'model-multimodal-clstm-inception',
                              cnnModelName = 'model-ucf101-optflow-inception',
                              trainListPath = '../splits/multimodal/trainlist01.txt',
                              testListPath  = '../splits/multimodal/testlist01.txt',
                              lblFilename  = '../classIndMulti.txt',
                              imuShape     = (30, 19),
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