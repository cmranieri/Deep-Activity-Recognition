import numpy as np
import os

from BaseTemporal import BaseTemporal

from keras.applications.inception_v3 import InceptionV3 as BaseModel
#from keras.applications.mobilenet import MobileNet as BaseModel
from keras.layers import Input, Dense, LSTM
from keras.layers import concatenate, Reshape, Permute
from keras.optimizers import SGD
from keras.models import Model



class TemporalLSTM( BaseTemporal ):
    
    def __init__( self,
                  restoreModel = True,
                  dim = 224,
                  timesteps = 8,
                  classes   = 101,
                  batchSize = 16,
                  rootPath  = '/home/olorin/Documents/caetano/datasets/UCF-101_flow',
                  modelPath = '/media/olorin/Documentos/caetano/ucf101/models',
                  modelName = 'model-lstm-final',
                  numThreads = 2,
                  maxsizeTrain = 4,
                  maxsizeTest  = 4,
                  lblFilename  = '../classInd.txt',
                  splitsDir    = '../splits/ucf101' ):
        super( TemporalLSTM , self ).__init__( restoreModel = restoreModel,
                                               dim = dim,
                                               timesteps    = timesteps,
                                               classes      = classes,
                                               batchSize    = batchSize,
                                               rootPath     = rootPath,
                                               modelPath    = modelPath,
                                               modelName    = modelName,
                                               numThreads   = numThreads,
                                               maxsizeTrain = maxsizeTrain,
                                               maxsizeTest  = maxsizeTest,
                                               lblFilename  = lblFilename,
                                               splitsDir    = splitsDir )


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
        
        y = LSTM( units = 128,
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
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    network = TemporalLSTM( restoreModel = False )
    #network.evaluate()
    network.train( epochs = 800000 )
