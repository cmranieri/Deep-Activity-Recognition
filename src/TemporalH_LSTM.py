import numpy as np
import os

from TemporalH import TemporalH

from keras.applications.inception_v3 import InceptionV3 as BaseModel
#from keras.applications.mobilenet import MobileNet as BaseModel
#from keras.applications.inception_resnet_v2 import InceptionResNetV2 as BaseModel
#from keras.applications.resnet50 import ResNet50 as BaseModel
#from keras.applications.mobilenet_v2 import MobileNetV2 as BaseModel
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
        y = LSTM( units = 64,
                  return_sequences = False,
                  dropout = 0.3,
                  unroll = True )( inp )
        y = Dense( self._classes, activation='softmax' )( y )
        
        model = Model( inp, y )
        optimizer = SGD( lr = 1e-2,
                         momentum = 0.9,
                         nesterov = True,
                         decay = 1e-5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model



if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalH_LSTM( dataDir      = '/home/cmranieri/datasets/UCF-101_flow',
                              modelDir     = '/home/cmranieri/models/ucf101',
                              modelName    = 'model-ucf101-hlstm-inception',
                              cnnModelName = 'model-ucf101-optflow-inception',
                              restoreModel = False,
                              normalize    = False )

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5 )
    network.train( steps      = 800000,
                   batchSize  = 16,
                   numThreads = 2,
                   maxsize    = 8 )
