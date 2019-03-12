import os
import numpy as np

datadir = '/media/olorin/Documentos/caetano/multimodal_dataset'
trainfiletxt = open( '../../splits/multimodal_dataset/trainlist.txt' , 'w' )
testfiletxt  = open( '../../splits/multimodal_dataset/testlist.txt'  , 'w' )
trainlist = list()
testlist  = list()

classes = sorted( os.listdir( os.path.join( datadir , 'video' ) ) )
for c in classes:
    files = sorted( os.listdir( os.path.join( datadir , 'video' , c ) ) )
    for f in files[ 0 : 7 ]:
        trainlist.append( os.path.join ( c , f ) )
        trainfiletxt.write( os.path.join( c , f ) + '\n' )
    for f in files[ 7 : ]:
        testlist.append( os.path.join( c , f ) )
        testfiletxt.write(  os.path.join( c , f ) + '\n' )
np.save( '../../splits/multimodal_dataset/trainlist.npy', trainlist )
np.save( '../../splits/multimodal_dataset/testlist.npy', testlist )
trainfiletxt.close()
testfiletxt.close()
