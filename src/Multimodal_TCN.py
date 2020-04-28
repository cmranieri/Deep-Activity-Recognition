import numpy as np
import os

from TemporalH import TemporalH

from keras.layers import Input, Dense, LSTM
from keras.layers import concatenate, Reshape, Permute
from keras.optimizers import SGD
from keras.models import Model
from tcn import TCN

class Multimodal_TCN( TemporalH ):
    def __init__( self, imuShape, **kwargs ):
        self.imuShape = imuShape
        super( Multimodal_TCN , self ).__init__( streams = ['temporal','inertial'],
                                                 **kwargs )


    def defineNetwork( self ):
        numFeatsFlow = int( self.cnnModel.output.shape[1] )
        # [ b, f, t ]
        flowInp = Input( shape = ( self.flowSteps, numFeatsFlow ) )
        imuModel = self.imuBlock( self.imuShape )
        merge = concatenate( [ flowInp, imuModel.outputs[0] ] )
        # [ b, t, f ]
        y = Permute( (2, 1) )( merge )
        y = TCN( nb_filters       = 128,
                 nb_stacks        = 3,
                 kernel_size      = 3,
                 use_skip_connections = True,
                 return_sequences = False,
                 dropout_rate     = 0.7,
                 dilations        = [ 1, 2, 4 ] )( merge )
        y = Dense( self.classes, activation='softmax' )( y )
        
        model = Model( [ flowInp, imuModel.inputs[0] ], y )
        
        optimizer = SGD( lr = 1e-2,
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
