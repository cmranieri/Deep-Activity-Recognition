import os

from BaseTemporal import BaseTemporal

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense 
from keras.optimizers import SGD
from keras.models import Model



class Temporal( BaseTemporal ):
    
    def __init__( self,
                  restoreModel = False,
                  dim = 224,
                  timesteps = 10,
                  classes   = 101,
                  batchSize = 32,
                  rootPath  = '/home/olorin/Documents/caetano/datasets/UCF-101_flow',
                  modelPath =  '/media/olorin/Documentos/caetano/ucf101/models',
                  modelName = 'model-norm',
                  numThreads = 4,
                  maxsizeTrain = 8,
                  maxsizeTest  = 6):
        super( Temporal , self ).__init__( restoreModel = restoreModel,
                                           dim = dim,
                                           timesteps    = timesteps,
                                           classes      = classes,
                                           batchSize    = batchSize,
                                           rootPath     = rootPath,
                                           modelPath    = modelPath,
                                           modelName    = modelName,
                                           numThreads   = numThreads,
                                           maxsizeTrain = maxsizeTrain,
                                           maxsizeTest  = maxsizeTest )



    def _defineNetwork( self ):
        input_tensor = Input( shape = ( self._dim, self._dim,
                                        2 * self._timesteps ) )
        model = InceptionV3( input_tensor = input_tensor,
                             weights = None,
                             classes = self._classes )
        optimizer = SGD( lr = 1e-2, momentum = 0.9,
                         nesterov=True, decay = 1e-5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    network = Temporal()
    #network.evaluate()
    network.train()
