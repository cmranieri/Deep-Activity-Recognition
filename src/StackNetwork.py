import os

from NetworkBase import NetworkBase

from tensorflow.keras.applications.inception_v3 import InceptionV3 as BaseModel
from tensorflow.keras.layers import Input 
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.optimizers.schedules import ExponentialDecay



class StackNetwork( NetworkBase ):
    
    def __init__( self, **kwargs ):
        super( StackNetwork , self ).__init__( streams = ['temporal'], **kwargs )


    def _prepareBatch( self, batchDict ):
        return batchDict[ 'temporal' ]
        


    def defineNetwork( self ):
        print( 'define network' )
        input_tensor = Input( shape = ( self.dim, self.dim,
                                        self.nFlowMaps * self.flowSteps ) )
        model = BaseModel( input_tensor = input_tensor,
                           weights = None,
                           classes = self.classes )

        #initial_learning_rate = 1e-2
        #lr_schedule = ExponentialDecay( initial_learning_rate,
        #                                decay_steps = 2000,
        #                                decay_rate  = 0.96,
        #                                staircase   = True )
        optimizer = SGD( lr=1e-2, momentum = 0.9, decay=1e-4 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics   = [ 'acc' ] ) 
        return model
        


if __name__ == '__main__':
    #os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = TemporalStack( flowDataDir  = '/lustre/cranieri/datasets/UCF-101_flow',
                             modelDir     = '/lustre/cranieri/models/ucf101',
                             modelName    = 'model-ucf101-optflow-inception2',
                             lblFilename  = '../classes/classInd.txt',
                             flowSteps    = 1,
                             clipTh       = 20,
                             framePeriod  = 1,
                             restoreModel = False,
                             normalize    = False )
    network.model.load_weights('/lustre/cranieri/models/ucf101/woptflow.h5')
    network.model.save('/lustre/cranieri/models/multimodal/model-ucf101-optflow-inception-multibackend.h5')
    #network.evaluate( numSegments  = 25,
    #                  smallBatches = 5,
    #                  storeTests   = True )
    #network.train( steps      = 200000,
    #               batchSize  = 32,
    #               numThreads = 12,
    #               maxsize    = 36 )
