import Spatial
import os
import sys

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'

split = sys.argv[1]

network = Spatial.Spatial( restoreModel = True,
                           classes      = 20,
                           dataDir      = '/home/cmranieri/datasets/multimodal_dataset_rgb',
                           modelDir     = '/home/cmranieri/models/multimodal',
                           modelName    = 'model-multi-spatial-l' + split[1],
                           lblFilename  = '../classIndMulti.txt',
                           numSegments  = 5,
                           smallBatches = 1,
                           splitsDir    = '../splits/multimodal_10',
                           split_n      = split )

network.train( epochs = 40000 )
#network.evaluate()
