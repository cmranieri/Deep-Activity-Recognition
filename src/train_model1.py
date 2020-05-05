from TemporalH_LSTM import TemporalH_LSTM as Network
import os
import sys

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( flowDataDir  = '/home/cmranieri/datasets/lyell/flow/',
                   imuDataDir   = '/home/cmranieri/datasets/lyell/inertial',
                   modelDir     = '/home/cmranieri/models/lyell/',
                   modelName    = 'model-lyell-vlstm-%s' % split_f,
                   cnnModelName = 'model-ucf101-optflow-inception',
                   trainListPath = '../splits/lyell/trainlist%s.txt' % split_f,
                   testListPath  = '../splits/lyell/testlist%s.txt' % split_f,
                   lblFilename  = '../classes/classIndLyell.txt',
                   imuShape     = ( 50, 6 ),
                   classes      = 9,
                   flowSteps    = 25,
                   imuSteps     = 50,
                   adjust       = False,
                   useFlips     = False,
                   framePeriod  = 2,
                   clipTh       = 20,
                   restoreModel = False,
                   normalize    = False )

network.train( steps        = 40000,
               stepsToEval  = 20000, 
               batchSize    = 16, 
               numThreads   = 4,
               maxsize      = 16,
               evalPer      = False )

network.evaluate( numSegments  = 25,
                  storeTests   = True )
