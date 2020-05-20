import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue
from InertialLoader import InertialLoader

class DataProvider:

    def __init__( self,
                  namesFilePath,
                  lblFilename,
                  flowDataDir = '',
                  rgbDataDir  = '',
                  imuDataDir  = '',
                  classes     = 101,
                  dim         = 224,
                  flowSteps   = 1,
                  imuSteps    = 20,
                  framePeriod = 1,
                  clipTh      = 20,
                  numThreads  = 1,
                  maxsize     = 10,
                  useFlips    = True,
                  normalize   = False,
                  ranges      = True ):
        self.lblFilename = lblFilename
        self.flowDataDir = flowDataDir
        self.rgbDataDir  = rgbDataDir
        self.imuDataDir  = imuDataDir
        self.classes     = classes
        self.dim         = dim
        self.flowSteps   = flowSteps
        self.imuSteps    = imuSteps
        self._numThreads = numThreads
        self._normalize  = normalize
        self._ranges     = ranges
        self.clipTh      = clipTh
        self.framePeriod = framePeriod
        self.useFlips    = useFlips
        
        self.filenames = self._loadFileNames( namesFilePath )
        self._length   = self.filenames.shape[ 0 ]
        self._reset()
        self._generateLabelsDict( lblFilename )
        self.imuDict = self.loadImuData( dataDir = imuDataDir )
        self._produce = True
        self._batchQueue = queue.Queue( maxsize = maxsize )
        self._threadsList = list()
 

    def __enter__( self ):
        self._startThreads()
        return self


    def __exit__( self, exc_type, exc_value, traceback ):
        self._produce = False
        for i, t in enumerate( self._threadsList ):
            t.join( 1 )
            print( 'Finished thread %d' % ( i ) )
        while not self._batchQueue.empty():
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


    def _loadFileNames( self, namesFilePath ):
        namesList = list()
        with open( namesFilePath, 'r' ) as f:
            for line in f:
                namesList.append( line.split(' ')[0].strip('\n') )
        return np.array( namesList )


    def loadImuData( self, dataDir ):
        if dataDir == '':
            return None
        imuDict = InertialLoader().load_data( data_dir = dataDir,
                                              classInd = self.lblFilename )
        return imuDict


    def provideRgbFrame( self, video, index ):
        frame = np.asarray( Image.open( video[ index ] ),
                            dtype = 'float32' )
        frame = frame[ ... , [ 2 , 1 , 0 ] ]
        frame = frame / 255.0
        return frame


    def provideFlowFrame( self, video, index, flowVecs = [ 'u', 'v' ] ):
        data = dict()
        ranges = dict()
        for vec in flowVecs:
            data[ vec ] = np.asarray( Image.open( video [ vec ] [index] ),
                                      dtype = 'float32' ) 
            if self._ranges:
                ranges[ vec ] = video ['%s_range'%vec] [index]
                cv2.normalize( data[ vec ],
                               data[ vec ],
                               ranges[ vec ][ 0 ],
                               ranges[ vec ][ 1 ],
                               cv2.NORM_MINMAX )
            if self.clipTh is not None and len( flowVecs )==2:
                data[ vec ][ data[ vec ] >  self.clipTh ] = self.clipTh
                data[ vec ][ data[ vec ] < -self.clipTh ] = -self.clipTh
            if self._normalize:
                data[ vec ] = data[ vec ] / max( np.max( np.abs( data[ vec ] ) ) , 1e-4 )
        ret = [ data[ key ] for key in data.keys() ]
        return ret


    def stackImu( self, key, start ):
        #start = min( start, len( self.imuDict[key] ) - self.imuSteps )
        window = self.imuDict[ key ][ start : start + self.imuSteps ]
        # [ t, f ]
        return np.array( window )


    def stackFlow( self, video, start ):
        flowList = list()
        for i in range( start,
                        start + self.flowSteps * self.framePeriod,
                        self.framePeriod ):
            u, v = self.provideFlowFrame( video, i )
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



