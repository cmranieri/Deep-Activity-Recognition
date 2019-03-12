import numpy as np
import pickle

a = 2
b = 1
out_a = pickle.load( open ('../outputs/model-norm.pickle', 'rb') )
out_b = pickle.load( open ('../outputs/model-ucf101-spatial.pickle', 'rb') )
correct_list = list()

for i in range( len( out_a['labels'] ) ):
    pred = a * out_a['predictions'][i] + b * out_b['predictions'][i]
    label = out_a['labels'][i]
    correct = np.equal( np.argmax( pred ),
                        np.argmax( label ) )
    correct_list.append(correct)

print( np.mean( correct_list ) )
