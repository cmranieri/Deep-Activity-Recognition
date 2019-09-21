import os

from NetworkBase import NetworkBase

from tensorflow.keras.applications.inception_v3 import InceptionV3 as BaseModel
from tensorflow.keras.layers import Input 
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.optimizers.schedules import ExponentialDecay



class TemporalStack( NetworkBase ):
    
    def __init__( self, **kwargs ):
        super( TemporalStack , self ).__init__( streams = ['temporal'], **kwargs )


    def _prepareBatch( self, batchDict ):
        return batchDict[ 'temporal' ]
        


    def defineNetwork( self ):
        input_tensor = Input( shape = ( self.dim, self.dim,
                                        2 * self.flowSteps ) )
        model = BaseModel( input_tensor = input_tensor,
                           weights = None,
                           classes = self.classes )

        #initial_learning_rate = 1e-2
        #lr_schedule = ExponentialDecay( initial_learning_rate,
        #                                decay_steps = 2000,
        #                                decay_rate  = 0.96,
        #                                staircase   = True )
        optimizer = SGD( lr=1e-2, momentum = 0.9, nesterov=True, decay=1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalStack( flowDataDir  = '/home/cmranieri/datasets/UCF-101_flow',
                             modelDir     = '/home/cmranieri/models/ucf101',
                             modelName    = 'model-ucf101-stack-inception',
                             flowSteps    = 15,
                             clipTh       = 20,
                             framePeriod  = 1,
                             restoreModel = True,
                             normalize    = False )

    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5,
    #                  storeTests   = True )
    network.train( steps      = 200000,
                   batchSize  = 32,
                   numThreads = 12,
                   maxsize    = 36 )
