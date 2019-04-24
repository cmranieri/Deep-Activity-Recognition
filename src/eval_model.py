from Spatial import Spatial as Network
import os
import sys

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'

split = int( sys.argv[1] )
split_f = '{:02d}'.format( split )

network = Network( restoreModel = True,
                   classes      = 20,
                   dataDir      = '/home/cmranieri/datasets/multimodal_dataset_rgb',
                   modelDir     = '/home/cmranieri/models/ucf101',
                   modelName    = 'model-ucf101-spatial_tl_multi-l'+split_f,
                   lblFilename  = '../classIndMulti.txt',
                   splitsDir    = '../splits/multimodal_10',
                   split_n      = split_f,
                   tl           = False,
                   tlSuffix     = '_tl_multi-l' + split_f )

#network.train( steps        = 20000,
#               batchSize    = 32, 
#               numThreads   = 4 )

network.evaluate( numSegments  = 25,
                  smallBatches = 5,
                  storeTests   = True )
