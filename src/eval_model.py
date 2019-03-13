from TemporalLSTM import TemporalLSTM as Network
import os
import sys

#os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'

#split = int( sys.argv[1] )
#split_f = '{:02d}'.format( split )

network = Network( restoreModel = True,
                   classes      = 20,
                   dataDir      = '/lustre/cranieri/datasets/UCF-101_flow',
                   modelDir     = '/lustre/cranieri/models/ucf101',
                   modelName    = 'model-ucf101-lstm',# + str(split),
                   lblFilename  = '../classInd.txt',
                   splitsDir    = '../splits/ucf101',
                   split_n      = '01' )#split_f )

#network.train( epochs = 40000 )
network.evaluate( numSegments = 5,
                  smallBatches = 1,
                  storeTests = False )
