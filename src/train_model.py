from Spatial import Spatial as Network
import os
import sys

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( restoreModel = True,
                   classes      = 20,
                   dataDir      = '/home/cmranieri/datasets/multimodal_dataset_rgb',
                   modelDir     = '/home/cmranieri/models/multimodal',
                   modelName    = 'model-multi-spatial-l' + str(split),
                   lblFilename  = '../classIndMulti.txt',
                   splitsDir    = '../splits/multimodal_10',
                   split_n      = split_f )

network.train( steps        = 40000,
               numThreads   = 2 )
#network.evaluate()
