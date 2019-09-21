from Multimodal_LSTM import Multimodal_LSTM as Network
import os
import sys

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( flowDataDir  = '/home/cmranieri/datasets/multimodal_dataset_flow',
                   imuDataDir   = '/home/cmranieri/datasets/multimodal_dataset_imu',
                   modelDir     = '/home/cmranieri/models/multimodal',
                   modelName    = 'model-multi-clstm-%s' % split_f,
                   cnnModelName = 'model-ucf101-optflow-inception',
                   trainListPath = '../splits/multimodal_10/trainlist%s.txt' % split_f,
                   testListPath  = '../splits/multimodal_10/testlist%s.txt' % split_f,
                   lblFilename  = '../classIndMulti.txt',
                   imuShape     = ( 30, 19 ),
                   classes      = 20,
                   flowSteps    = 15,
                   imuSteps     = 30,
                   framePeriod  = 4,
                   clipTh       = 20,
                   restoreModel = False,
                   normalize    = False )

network.train( steps        = 20000,
               stepsToEval  = 30000, 
               batchSize    = 16, 
               numThreads   = 4,
               maxsize      = 12 )

network.evaluate( numSegments  = 25,
                  storeTests   = True )
