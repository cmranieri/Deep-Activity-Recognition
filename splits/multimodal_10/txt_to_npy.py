import numpy as np

for fold in range(10):
    fold_id = '{:02d}'.format(fold+1)

    sfile = open( 'trainlist'+fold_id+'.txt' , 'r' )
    dlist = list()
    for line in sfile:
        dlist.append( line.strip('\n') )
    np.save( 'trainlist'+fold_id+'.npy' , np.array(dlist) )
    sfile.close()

    sfile = open( 'testlist' +fold_id+'.txt' , 'r' )
    dlist = list()
    for line in sfile:
        dlist.append( line.strip('\n') )
    np.save( 'testlist' +fold_id+'.npy' , np.array(dlist) )
    sfile.close()
