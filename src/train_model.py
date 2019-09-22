from Multimodal_LSTM import Multimodal_LSTM as Network
import os
import sys

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( flowDataDir  = '/home/cmranieri/datasets/UTD-MHAD/flow',
                   imuDataDir   = '/home/cmranieri/datasets/UTD-MHAD/Inertial_csv',
                   modelDir     = '/home/cmranieri/models/utd-mhad',
                   modelName    = 'model-utd-clstm-%s' % split_f,
                   cnnModelName = 'model-ucf101-optflow-inception',
                   trainListPath = '../splits/utd-mhad/trainlist%s.txt' % split_f,
                   testListPath  = '../splits/utd-mhad/testlist%s.txt' % split_f,
                   lblFilename  = '../classes/classIndUtd.txt',
                   imuShape     = ( 30, 6 ),
                   classes      = 27,
                   flowSteps    = 15,
                   imuSteps     = 30,
                   framePeriod  = 2,
                   clipTh       = 20,
                   restoreModel = False,
                   normalize    = False )

network.train( steps        = 20000,
               stepsToEval  = 30000, 
               batchSize    = 16, 
               numThreads   = 4,
               maxsize      = 12 )

network.evaluate( numSegments  = 2,
                  storeTests   = True )
