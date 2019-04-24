import numpy as np
import pickle


a = 0
b = 1
c = 0

acc_list = list()
for j in range(1):
    split_f = '{:02d}'.format( j+1 )
    #out_a = pickle.load( open ('../outputs/model-ucf101-spatial-mobilenet.pickle', 'rb') )
    #out_b = pickle.load( open ('../outputs/model-ucf101-stack-inres.pickle', 'rb') )
    out_a = pickle.load( open ('../outputs/model-multi-spatial-m'+str(j+1)+'.pickle', 'rb') )
    out_b = pickle.load( open ('../outputs/model-ucf101-stack-inception_tl_multi-l'+split_f+'.pickle', 'rb') )
    #out_c = pickle.load( open ('../outputs/caetano-cnn-lstm-l'+str(j+1)+'.pickle', 'rb') )
    correct_list = list()

    for i in range( len( out_b['labels'] ) ):
        pred = a * out_a['predictions'][i] + b * out_b['predictions'][i] #+ c * out_c['predictions'][i]
        #pred = out_b['predictions'][i] #+ c * out_c['predictions'][i]
        label = out_b['labels'][i]
        correct = np.equal( np.argmax( pred ),
                            np.argmax( label ) )
        correct_list.append(correct)

    acc_list.append( np.mean( correct_list ) )

print(np.mean(acc_list))
