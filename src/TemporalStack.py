import os

from NetworkBase import NetworkBase

from keras.applications.inception_v3 import InceptionV3 as BaseModel
#from keras.applications.mobilenet import MobileNet as BaseModel
#from keras.applications.inception_resnet_v2 import InceptionResNetV2 as BaseModel
#from keras.applications.resnet50 import ResNet50 as BaseModel
#from keras.applications.mobilenet_v2 import MobileNetV2 as BaseModel
from keras.layers import Input, Dense 
from keras.optimizers import SGD, Adam
from keras.models import Model



class TemporalStack( NetworkBase ):
    
    def __init__( self, **kwargs ):
        super( TemporalStack , self ).__init__( stream = 'temporal', **kwargs )


    def _defineNetwork( self ):
        input_tensor = Input( shape = ( self._dim, self._dim,
                                        2 * self._timesteps ) )
        model = BaseModel( input_tensor = input_tensor,
                           weights = None,
                           classes = self._classes )
        optimizer = SGD( lr = 1e-2, momentum = 0.9,
                         nesterov = True, decay = 1e-5 )
        # optimizer = Adam ( lr = 1e-2, decay = 1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    network = TemporalStack( dataDir      = '/lustre/cranieri/datasets/UCF-101_flow',
                             modelDir     =  '/lustre/cranieri/models/ucf101',
                             modelName    = 'model-ucf101-optflow-inception',
                             timesteps    = 1,
                             restoreModel = False,
                             normalize    = False )

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5,
    #                  storeTests   = True )
    network.train( steps     = 800000,
                   batchSize = 64,
                   numThreads = 12,
                   maxsize   = 32 )
