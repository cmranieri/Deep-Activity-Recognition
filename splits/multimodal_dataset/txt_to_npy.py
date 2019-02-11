import numpy as np

sfile = open( 'trainlist03.txt' , 'r' )

dlist = list()
for line in sfile:
    dlist.append( line.strip('\n') )
np.save( 'trainlist03.npy' , np.array(dlist) )
