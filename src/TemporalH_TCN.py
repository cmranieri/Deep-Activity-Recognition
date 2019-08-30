import numpy as np
import os

from TemporalH import TemporalH

from keras.applications.inception_v3 import InceptionV3 as BaseModel
#from keras.applications.mobilenet import MobileNet as BaseModel
#from keras.applications.inception_resnet_v2 import InceptionResNetV2 as BaseModel
#from keras.applications.mobilenet_v2 import MobileNetV2 as BaseModel
from keras.layers import Input, Dense 
from keras.layers import concatenate, Reshape, Permute
from keras.optimizers import SGD
from keras.models import Model
from tcn import TCN



class TemporalH_TCN( TemporalH ):
    def __init__( self, **kwargs ):
        super( TemporalH_TCN , self ).__init__( **kwargs )


   
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
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalTCN( dataDir      = '/lustre/cranieri/datasets/UCF-101_flow',
                           modelDir     = '/lustre/cranieri/models/ucf101',
                           modelName    = 'model-ucf101-tcn-inception',
                           restoreModel = False,
                           normalize    = False)

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5,
    #                  storeTests   = True )
    network.train( steps      = 800000,
                   batchSize  = 16,
                   numThreads = 8,
                   maxsize    = 24 )  
