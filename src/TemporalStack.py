import os

from NetworkBase import NetworkBase

from tensorflow.keras.applications.inception_v3 import InceptionV3 as BaseModel
from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Model



class TemporalStack( NetworkBase ):
    
    def __init__( self, **kwargs ):
        super( TemporalStack , self ).__init__( stream = 'temporal', **kwargs )


    def _defineNetwork( self ):
        input_tensor = Input( shape = ( self.dim, self.dim,
                                        2 * self.timesteps ) )
        model = BaseModel( input_tensor = input_tensor,
                           weights = None,
                           classes = self.classes )

        initial_learning_rate = 1e-2
        lr_schedule = ExponentialDecay( initial_learning_rate,
                                        decay_steps = 4000,
                                        decay_rate  = 0.92,
                                        staircase   = True )
        optimizer = SGD( learning_rate = lr_schedule, momentum = 0.9 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    network = TemporalStack( dataDir      = '/lustre/cranieri/datasets/UCF-101_flow',
                             modelDir     = '/lustre/cranieri/models/ucf101',
                             modelName    = 'model-ucf101-optflow-inception',
                             timesteps    = 1,
                             clipTh       = 20,
                             framePeriod  = 2,
                             restoreModel = False,
                             normalize    = False )

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5,
    #                  storeTests   = True )
    network.train( steps     = 400000,
                   batchSize = 16,
                   numThreads = 12,
                   maxsize   = 32 )
