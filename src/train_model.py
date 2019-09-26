from Multimodal_TCN import Multimodal_TCN as Network
import os
import sys

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( flowDataDir  = '/home/cmranieri/datasets/UTD-MHAD/flow',
                   imuDataDir   = '/home/cmranieri/datasets/UTD-MHAD/Inertial_csv',
                   modelDir     = '/home/cmranieri/models/utd-mhad',
                   modelName    = 'model-utd-ctcn-%s' % split_f,
                   cnnModelName = 'model-ucf101-optflow-inception',
                   trainListPath = '../splits/utd-mhad/trainlist%s.txt' % split_f,
                   testListPath  = '../splits/utd-mhad/testlist%s.txt' % split_f,
                   lblFilename  = '../classes/classIndUtd.txt',
                   imuShape     = ( 100, 6 ),
                   classes      = 27,
                   useFlips     = False,
                   flowSteps    = 15,
                   imuSteps     = 100,
                   adjust       = True,
                   framePeriod  = 2,
                   clipTh       = 20,
                   restoreModel = False,
                   normalize    = False )

network.train( steps        = 30000,
               stepsToEval  = 30000, 
               batchSize    = 16, 
               numThreads   = 3,
               maxsize      = 12,
               evalPer      = False )

network.evaluate( numSegments  = 2,
                  storeTests   = True )
