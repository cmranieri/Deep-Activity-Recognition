import TemporalLSTM

network = TemporalLSTM.TemporalLSTM( restoreModel = False,
                                     classes   = 20,
                                     rootPath  = '/home/cranieri/datasets/multimodal_dataset_flow',
                                     modelPath = '/lustre/cranieri/models',
                                     modelName = 'model-lstm-multi-video',
                                     lblFilename = '../classIndMulti.txt',
                                     splitsDir   = '../splits/multimodal_dataset')

network.train( epochs = 500000 )
