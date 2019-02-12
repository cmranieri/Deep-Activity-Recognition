import Temporal
import os

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'

network = Temporal.Temporal( restoreModel = False,
                                     classes     = 20,
                                     rootPath    = '/home/cmranieri/datasets/multimodal_dataset_flow2',
                                     modelPath   = '/home/cmranieri/models/ucf101',
                                     modelName   = 'model-norm',
                                     lblFilename = '../classIndMulti.txt',
                                     splitsDir   = '../splits/multimodal_dataset',
                                     split_n     = '02',
                                     tl          = True,
                                     tlSuffix    = '_tl_multi-2')

network.train( epochs = 100000 )
