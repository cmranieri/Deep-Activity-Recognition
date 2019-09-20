from TemporalStack import TemporalStack as Network
import os
import sys

#os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

#split = int( sys.argv[1] )
#split_f = '{:02d}'.format( split )

network = Network( restoreModel  = True,
                   classes       = 101,
                   flowDataDir   = '/lustre/cranieri/datasets/UCF-101_flow',
                   flowSteps     = 1,
                   modelDir      = '/lustre/cranieri/models/ucf101',
                   modelName     = 'model-ucf101-optflow-inception',
                   lblFilename   = '../classInd.txt',
                   trainListPath = '../splits/ucf101/trainlist01.txt',
                   testListPath  = '../splits/ucf101/testlist01.txt',
                   tl            = False )
                   #tlSuffix      = '_tl_multi-l' + split_f )

#network.train( steps        = 20000,
#               batchSize    = 32, 
#               numThreads   = 4 )

network.evaluate( numSegments  = 25,
                  storeTests   = True )
