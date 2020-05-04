from TemporalH_TCN import TemporalH_TCN as Network
import os
import sys

#os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( flowDataDir  = '/lustre/cranieri/datasets/lyell/flow/',
                   imuDataDir   = '/lustre/cranieri/datasets/lyell/inertial',
                   modelDir     = '/lustre/cranieri/models/lyell/',
                   modelName    = 'model-lyell-vtcn-%s' % split_f,
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
                   framePeriod  = 1,
                   clipTh       = 20,
                   restoreModel = False,
                   normalize    = False )

network.train( steps        = 30000,
               stepsToEval  = 30000, 
               batchSize    = 16, 
               numThreads   = 12,
               maxsize      = 36,
               evalPer      = False )

network.evaluate( numSegments  = 5,
                  storeTests   = True )
