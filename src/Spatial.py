import os

from NetworkBase import NetworkBase

from keras.applications.inception_v3 import InceptionV3 as BaseModel
from keras.layers import Input, Dense 
from keras.optimizers import SGD
from keras.models import Model


class Spatial( NetworkBase ):
    
    def __init__( self, **kwargs ):
        super( Spatial , self ).__init__( stream = 'spatial', **kwargs )


    def _defineNetwork( self ):
        input_tensor = Input( shape = ( self._dim, self._dim, 3 ) )
        base_model = BaseModel( input_tensor = input_tensor,
                                weights = 'imagenet',
                                include_top = False,
                                pooling = 'avg' )
        y = base_model.output
        y = Dense( self._classes, activation='softmax' )( y )
        
        model = Model( inputs = base_model.input, outputs = y )
        for layer in base_model.layers:
                layer.trainable = False

        optimizer = SGD( lr = 1e-2, momentum = 0.9,
                         nesterov=True, decay = 1e-5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    network = Spatial( rgbDataDir   = '/home/cmranieri/datasets/UCF-101_rgb',
                       modelDir     = '/home/cmranieri/models/ucf101',
                       modelName    = 'model-ucf101-spatial-inception',
                       restoreModel = False,
                       normalize    = False )

    #network.train( steps = 10000,
    #               batchSize = 32,
    #               numThreads = 2 )   

    network.evaluate( numSegments  = 25,
                      smallBatches = 5,
                      storeTests   = True )

