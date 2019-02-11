import TemporalTCN
import os

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

network = TemporalTCN.TemporalTCN( restoreModel = False,
                                     classes     = 20,
                                     rootPath    = '/home/cmranieri/datasets/multimodal_dataset_flow2',
                                     modelPath   = '/home/cmranieri/models/multimodal',
                                     modelName   = 'model-multi-video-tcn-1',
                                     lblFilename = '../classIndMulti.txt',
                                     splitsDir   = '../splits/multimodal_dataset',
                                     split_n     = '01' )

network.train( epochs = 200000 )
