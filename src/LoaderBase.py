import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue

class LoaderBase:

    def __init__( self,
                  dataDir,
                  filenames,
                  lblFilename,
                  classes    = 101,
                  dim        = 224,
                  timesteps  = 10,
                  numThreads = 1,
                  maxsize    = 10,
                  normalize  = False,
                  clip       = True,
                  ranges     = True ):
        self.dataDir      = dataDir
        self.filenames    = filenames
        self._classes     = classes
        self.dim          = dim
        self._timesteps   = timesteps
        self._numThreads  = numThreads
        self._length      = filenames.shape[ 0 ]
        self._normalize   = normalize
        self._clip        = clip
        self._ranges      = ranges

        self.pxClipTh = 20
        self.framePeriod = 2
        
        self._reset()
        self._generateLabelsDict( lblFilename )

        self._produce = True
        self._batchQueue = queue.Queue( maxsize = maxsize )
        self._threadsList = list()
 

    def __enter__( self ):
        self._startThreads()
        return self


    def __exit__( self, exc_type, exc_value, traceback ):
        self._produce = False
        for i, t in enumerate( self._threadsList ):
            t.join(1)
            print( 'Finished thread %d' % ( i ) )
        if not self._batchQueue.empty():
            self._batchQueue.get()


    def _generateLabelsDict( self, filename ):
        self._labelsDict = dict()
        f = open( filename , 'r' )
        for line in f.readlines():
            self._labelsDict[ line.split()[ 1 ] ] = line.split()[ 0 ]

    def _reset( self ):
        raise NotImplementedError( 'Please implement this method' )


    def _incIndex( self ):
        raise NotImplementedError( 'Please implement this method' )


    def loadRgb( self, video, index ):
        frame = np.asarray( Image.open( video[ index ] ),
                            dtype = 'float32' )
        frame = frame[ ... , [ 2 , 1 , 0 ] ]
        frame = frame / 255.0
        return frame


    def loadFlow( self, video, index ):
        u = np.asarray( Image.open( video ['u'] [index] ) , dtype = 'float32' )
        v = np.asarray( Image.open( video ['v'] [index] ) , dtype = 'float32' )

        if self._ranges:
            u_range = video ['u_range'] [index]
            v_range = video ['v_range'] [index]
            cv2.normalize( u, u, u_range[0], u_range[1], cv2.NORM_MINMAX )
            cv2.normalize( v, v, v_range[0], v_range[1], cv2.NORM_MINMAX )

        if self._clip:
            u[ u > self.pxClipTh ] = self.pxClipTh
            v[ v > self.pxClipTh ] = self.pxClipTh

        if self._normalize:
            u = u / max( np.max( np.abs( u ) ) , 1e-4 ) 
            v = v / max( np.max( np.abs( v ) ) , 1e-4 ) 
        return u, v


    def stackFlow( self, video, start ):
        flowList = list()
        for i in range( start,
                        start + self._timesteps * self.framePeriod,
                        self.framePeriod ):
            u, v = self.loadFlow( video, i )
            flowList.append( np.array( [ u , v ] ) )
        stack = np.array( flowList )
        # [ u, v, 2, t ]
        stack = np.transpose( stack , [ 2 , 3 , 0 , 1 ] )
        return stack


    def _startThreads( self ):
        for i in range( self._numThreads ):
            print( 'Initializing thread %d' % ( i ) )
            t = Thread( target = self._batchThread )
            self._threadsList.append( t )
            t.start()


    def _batchThread( self ):
        raise NotImplementedError( 'Please implement this method' )

    def getBatch( self ):
        raise NotImplementedError( 'Please Implement this method' )



