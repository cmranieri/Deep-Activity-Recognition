import Temporal
import os

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'

network = Temporal.Temporal( restoreModel = False,
                                     classes   = 20,
                                     rootPath  = '/home/cmranieri/datasets/multimodal_dataset_flow2',
                                     modelPath = '/home/cmranieri/models/multimodal',
                                     modelName = 'model-multi-video',
                                     lblFilename = '../classIndMulti.txt',
                                     splitsDir   = '../splits/multimodal_dataset')

network.train( epochs = 800000 )
