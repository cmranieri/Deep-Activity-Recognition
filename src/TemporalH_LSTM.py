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
        inputs_list = list()
        feats_list  = list()

        for i in range( self._timesteps ):
            inp = Input( shape = (self._dim, self._dim, 2) )
            inputs_list.append( inp )
            base_model = BaseModel( input_tensor = inp,
                                    weights = None,
                                    include_top = False,
                                    pooling = 'avg' )
            for layer in base_model.layers:
                layer.name = layer.name + '_n' + str(i)
            feats_list.append( base_model.outputs[0] )

        merge = concatenate( feats_list )
        num_feats = int(merge.shape[ 1 ]) // self._timesteps
        merge = Reshape( [ num_feats , self._timesteps ] )( merge )
        merge = Permute( (2, 1) )( merge )
        
        y = LSTM( units = 64,
                  return_sequences = False,
                  dropout = 0.3,
                  unroll = True )( merge )
        y = Dense( self._classes, activation='softmax' )( y )
        
        model = Model( inputs_list, y )
        optimizer = SGD( lr = 1e-2,
                         momentum = 0.9,
                         nesterov = True,
                         decay = 1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model



    def _prepareBatch( self, batch ):
        batch = np.reshape( batch, [ batch.shape[0],
                                     self._dim, self._dim,
                                     2, self._timesteps ] )
        batch = list( np.transpose( batch, [ 4, 0, 1, 2, 3 ] ) )
        return batch



if __name__ == '__main__':
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalLSTM( dataDir      = '/lustre/cranieri/datasets/UCF-101_flow',
                            modelDir     = '/lustre/cranieri/models/ucf101',
                            modelName    = 'model-ucf101-lstm-inception',
                            restoreModel = False,
                            normalize    = False )

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5 )
    network.train( steps      = 800000,
                   batchSize  = 16,
                   numThreads = 8,
                   maxsize    = 24 )
