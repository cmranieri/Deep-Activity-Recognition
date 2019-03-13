import os

from NetworkBase import NetworkBase

#from keras.applications.inception_v3 import InceptionV3 as BaseModel
from keras.applications.mobilenet import MobileNet as BaseModel
from keras.layers import Input, Dense 
from keras.optimizers import SGD
from keras.models import Model



class TemporalStack( NetworkBase ):
    
    def __init__( self,
                  restoreModel = False,
                  dim          = 224,
                  timesteps    = 8,
                  classes      = 101,
                  dataDir      = '/home/cmranieri/datasets/UCF-101_flow',
                  modelDir     =  '/home/cmranieri/models/ucf101',
                  modelName    = 'model-ucf101-stack',
                  lblFilename  = '../classInd.txt',
                  splitsDir    = '../splits/ucf101',
                  split_n      = '01',
                  tl           = False,
                  tlSuffix     = '',
                  stream       = 'temporal',
                  normalize    = False):

        super( TemporalStack , self ).__init__( restoreModel = restoreModel,
                                                dim          = dim,
                                                timesteps    = timesteps,
                                                classes      = classes,
                                                dataDir      = dataDir,
                                                modelDir     = modelDir,
                                                modelName    = modelName,
                                                lblFilename  = lblFilename,
                                                splitsDir    = splitsDir,
                                                split_n      = split_n,
                                                tl           = tl,
                                                tlSuffix     = tlSuffix,
                                                stream       = stream,
                                                normalize    = normalize)



    def _defineNetwork( self ):
        input_tensor = Input( shape = ( self._dim, self._dim,
                                        2 * self._timesteps ) )
        model = BaseModel( input_tensor = input_tensor,
                           weights = None,
                           classes = self._classes )
        optimizer = SGD( lr = 1e-2, momentum = 0.9,
                         nesterov = True, decay = 1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalStack( restoreModel = False,
                             normalize = True )

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5,
    #                  storeTests   = True )
    network.train( steps = 800000,
                   batchSize = 16,
                   maxsize = 16 )
