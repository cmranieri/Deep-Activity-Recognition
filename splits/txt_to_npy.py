import numpy as np

d = 'multimodalb_10'

for fold in range(10):
    fold_id = '{:02d}'.format(fold+1)

    sfile = open( d + '/trainlist'+fold_id+'.txt' , 'r' )
    dlist = list()
    for line in sfile:
        dlist.append( line.strip('\n') )
    np.save( d + '/trainlist'+fold_id+'.npy' , np.array(dlist) )
    sfile.close()

    sfile = open( d + '/testlist' +fold_id+'.txt' , 'r' )
    dlist = list()
    for line in sfile:
        dlist.append( line.strip('\n') )
    np.save( d + '/testlist' +fold_id+'.npy' , np.array(dlist) )
    sfile.close()
