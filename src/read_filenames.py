import numpy as np


trainlist_file = open( '../splits/trainlist011.txt' , 'r' )
trainlist = list()
for line in trainlist_file.readlines():
    trainlist += [ line.split()[0] ]

np.save( '../splits/trainlist011.npy' , trainlist )


testlist_file = open( '../splits/testlist01.txt' , 'r' )
testlist = list()
for line in testlist_file.readlines():
    testlist += [ line.split()[0] ]

np.save( '../splits/testlist01.npy' , testlist )

