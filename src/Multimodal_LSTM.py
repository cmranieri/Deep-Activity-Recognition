import numpy as np
import os

from TemporalH2 import TemporalH

from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.layers import concatenate, Reshape, Permute
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model


class Multimodal_LSTM( TemporalH ):
    def __init__( self, imuShape, **kwargs ):
        self.imuShape = imuShape
        super( Multimodal_LSTM , self ).__init__( streams = ['temporal','inertial'],
                                                 **kwargs )


    def defineNetwork( self ):
        numFeatsFlow = int( self.cnnModel.output.shape[1] )
        # [ b, f, t ]
        flowInp = Input( shape = ( self.flowSteps, numFeatsFlow ) )
        imuModel = self.imuBlock( self.imuShape )
        merge = concatenate( [ flowInp, imuModel.outputs[0] ] )
        # [ b, t, f ]
        y = Permute( (2, 1) )( merge )
        y = LSTM( units             = 128,
                  return_sequences  = False,
                  dropout           = 0.1,
                  recurrent_dropout = 0.0,
                  unroll            = False )( merge )
        y = Dense( self.classes, activation='softmax' )( y )
        
        model = Model( [ flowInp, imuModel.inputs[0] ], y )
        optimizer = SGD( lr = 1e-2 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
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
