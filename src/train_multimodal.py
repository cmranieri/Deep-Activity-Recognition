import TemporalTCN
import os

os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

network = TemporalTCN.TemporalTCN( restoreModel = True,
                                   classes     = 20,
                                   dataDir    = '/home/cmranieri/datasets/multimodal_dataset_flow2',
                                   modelDir   = '/home/cmranieri/models/ucf101',
                                   modelName   = 'model-tcn-final_tl_multi-l7',
                                   lblFilename = '../classIndMulti.txt',
                                   splitsDir   = '../splits/multimodal_10',
                                   split_n     = '07',
                                   tl          = False,
                                   tlSuffix    = '_tl_multi-l1')

network.train( epochs = 80000 )
