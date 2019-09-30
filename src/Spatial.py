import os

from NetworkBase import NetworkBase

from tensorflow.keras.applications.inception_v3 import InceptionV3 as BaseModel
from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model


class Spatial( NetworkBase ):
    
    def __init__( self, **kwargs ):
        super( Spatial , self ).__init__( streams = ['spatial'], **kwargs )


    def _prepareBatch( self, batchDict ):
        return batchDict[ 'spatial' ]


    def defineNetwork( self ):
        input_tensor = Input( shape = ( self.dim, self.dim, 3 ) )
        base_model = BaseModel( input_tensor = input_tensor,
                                weights = 'imagenet',
                                include_top = False,
                                pooling = 'avg' )
        y = base_model.output
        y = Dense( self.classes, activation='softmax' )( y )
        
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
    
    network = Spatial( rgbDataDir   = '/mnt/sda2/datasets/UTD-MHAD/RGB_pickle',
                       modelDir     = '/mnt/sda2/models/',
                       modelName    = 'model-utd-spatial-inception',
                       trainListPath = '../splits/multimodal_10/trainlist%s.txt' % split_f,
                       testListPath  = '../splits/multimodal_10/testlist%s.txt' % split_f,
                       lblFilename  = '../classes/classIndUtd.txt',
                       classes = 27,
                       useFlips = False,
                       restoreModel = False,
                       normalize    = False )

    #network.train( steps = 10000,
    #               batchSize = 32,
    #               numThreads = 2 )   

    network.evaluate( numSegments  = 25,
                      storeTests   = True )

