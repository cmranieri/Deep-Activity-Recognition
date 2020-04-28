from Multimodal_TCN import Multimodal_TCN as Network
import os
import sys

#os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( flowDataDir  = '/lustre/cranieri/datasets/multimodal_dataset_flow',
                   imuDataDir   = '/lustre/cranieri/datasets/multimodal_dataset/sensor/',
                   modelDir     = '/lustre/cranieri/models/multimodal/',
                   modelName    = 'model-multi-ctcn3-%s' % split_f,
                   cnnModelName = 'model-ucf101-optflow-inception-multibackend',
                   trainListPath = '../splits/multimodal_10/trainlist%s.txt' % split_f,
                   testListPath  = '../splits/multimodal_10/testlist%s.txt' % split_f,
                   lblFilename  = '../classes/classIndMulti.txt',
                   imuShape     = ( 30, 19 ),
                   classes      = 20,
                   flowSteps    = 15,
                   imuSteps     = 30,
                   adjust       = False,
                   useFlips     = False,
                   framePeriod  = 4,
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
