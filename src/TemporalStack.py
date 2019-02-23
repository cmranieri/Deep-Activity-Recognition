import os

from NetworkBase import NetworkBase

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense 
from keras.optimizers import SGD
from keras.models import Model



class TemporalStack( NetworkBase ):
    
    def __init__( self,
                  restoreModel = False,
                  dim = 224,
                  timesteps = 8,
                  classes   = 101,
                  batchSize = 16,
                  rootPath  = '/home/cmranieri/datasets/UCF-101_flow',
                  modelPath =  '/home/cmranieri/models/ucf101',
                  modelName = 'model-stack',
                  numThreads = 2,
                  maxsizeTrain = 4,
                  maxsizeTest  = 4,
                  lblFilename  = '../classInd.txt',
                  splitsDir    = '../splits/ucf101',
                  split_n = '01',
                  tl = False,
                  tlSuffix = '' ):

        super( TemporalStack , self ).__init__( restoreModel = restoreModel,
                                                dim          = dim,
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
                                                splitsDir    = splitsDir,
                                                split_n      = split_n,
                                                tl           = tl,
                                                tlSuffix     = tlSuffix )



    def _defineNetwork( self ):
        input_tensor = Input( shape = ( self._dim, self._dim,
                                        2 * self._timesteps ) )
        model = InceptionV3( input_tensor = input_tensor,
                             weights = None,
                             classes = self._classes )
        optimizer = SGD( lr = 1e-2, momentum = 0.9,
                         nesterov=True, decay = 1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    network = TemporalStack( restoreModel = False )
    #network.evaluate()
    network.train( epochs = 600000 )
