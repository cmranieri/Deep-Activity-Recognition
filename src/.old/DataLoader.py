import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue

class DataLoader:

    def __init__( self,
                  rootPath,
                  filenames,
                  lblFilename ,
                  dim = 224,
                  timesteps = 10,
                  numThreads = 1,
                  maxsize = 10 ):
        self.rootPath    = rootPath
        self.filenames   = filenames
        self.dim         = dim
        self._timesteps  = timesteps
        self._numThreads = numThreads
        self._length      = filenames.shape[ 0 ]
        
        self._reset()
        self._generateLabelsDict( lblFilename )

        self._produce = True
        self._batchQueue = queue.Queue( maxsize = maxsize )
        self._indexMutex = Lock()
        self._queueMutex = Lock()
        self._threadsList = list()
 

    def __enter__( self ):
        self._startThreads()
        return self


    def __exit__( self, exc_type, exc_value, traceback ):
        self._produce = False
        for i, t in enumerate( self._threadsList ):
            t.join()
            print( 'Finished thread %d' % ( i ) )


    def _generateLabelsDict( self, filename ):
        self.labelsDict = dict()
        f = open( filename , 'r' )
        for line in f.readlines():
            self.labelsDict[ line.split()[ 1 ] ] = line.split()[ 0 ]

    def _reset( self ):
        raise NotImplementedError("Please Implement this method")


    def _incIndex( self ):
        raise NotImplementedError("Please Implement this method")


    def loadFlow( self, video, index ):
        u = np.asarray( Image.open( video ['u'] [index] ) , dtype = 'float32' )
        v = np.asarray( Image.open( video ['v'] [index] ) , dtype = 'float32' )
        u_range = video ['u_range'] [index]
        v_range = video ['v_range'] [index]

        #u = np.array( u_img, dtype = 'float32' ).copy()
        #v = np.array( v_img, dtype = 'float32' ).copy()
        cv2.normalize( u, u,  u_range[ 0 ], u_range[ 1 ], cv2.NORM_MINMAX )
        cv2.normalize( v, v,  v_range[ 0 ], v_range[ 1 ], cv2.NORM_MINMAX )

        u = u / max( np.max( np.abs( u ) ) , 0.001 )
        v = v / max( np.max( np.abs( v ) ) , 0.001 ) 
        return u, v


    def stackFlow( self, video, start ):
        flowList = list()
        for i in range( start, start + self._timesteps ):
            u, v = self.loadFlow( video, i )
            flowList.append( np.array( [ u , v ] ) )
        stack = np.array( flowList )
        # [ u, v, 2, t ]
        stack = np.transpose( stack , [ 2 , 3 , 1 , 0 ] )
        return stack


    def _startThreads( self ):
        for i in range( self._numThreads ):
            print( 'Initializing thread %d' % ( i ) )
            t = Thread( target = self._batchThread )
            self._threadsList.append( t )
            t.start()


    def _batchThread( self ):
        raise NotImplementedError("Please Implement this method")

    def getBatch( self ):
        raise NotImplementedError("Please Implement this method")



