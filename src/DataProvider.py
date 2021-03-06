import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue
from RawDataLoader import RawDataLoader

class DataProvider:

    def __init__( self,
                  namesFilePath,
                  lblFilename,
                  flowDataDir = '',
                  rgbDataDir  = '',
                  imuDataDir  = '',
                  homeDataDir = '',
                  imuClassDirs = True,
                  classes     = 101,
                  dim         = 224,
                  flowSteps   = 1,
                  imuSteps    = 20,
                  framePeriod = 1,
                  clipTh      = 20,
                  numThreads  = 1,
                  maxsize     = 10,
                  nFlowMaps   = 2,
                  scaleFlow   = False,
                  useFlips    = False,
                  normalize   = False,
                  ranges      = True ):
        self.lblFilename = lblFilename
        self.flowDataDir = flowDataDir
        self.rgbDataDir  = rgbDataDir
        self.imuDataDir  = imuDataDir
        self.imuClassDirs = imuClassDirs
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
        self.nFlowMaps   = nFlowMaps
        self.scaleFlow   = scaleFlow
        
        self.filenames = self._loadFileNames( namesFilePath )
        self._length   = self.filenames.shape[ 0 ]
        self._reset()
        self._generateLabelsDict( lblFilename )
        self.imuDict  = self.loadRawData( dataDir = imuDataDir )
        self.homeDict = self.loadRawData( dataDir = homeDataDir )
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


    def loadRawData( self, dataDir ):
        if dataDir == '':
            return None
        imuDict = RawDataLoader().load_data( data_dir  = dataDir,
                                             classInd  = self.lblFilename,
                                             diff_dirs = self.imuClassDirs )
        return imuDict


    def provideRgbFrame( self, video, index ):
        frame = np.asarray( Image.open( video[ index ] ),
                            dtype = 'float32' )
        #frame = frame[ ... , [ 2 , 1 , 0 ] ]
        frame = frame / 255.0
        return frame


    def provideFlowFrame( self, video, index ):
        if self.nFlowMaps==2:
            flowVecs = [ 'u', 'v' ]
        elif self.nFlowMaps==3:
            flowVecs = [ 'v', 'w', 'u' ]
            #flowVecs = [ 'u', 'v', 'w' ]
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
            if self.scaleFlow:
                data[ vec ] = data[ vec ] * 100
            if self._normalize:
                data[ vec ] = data[ vec ] / max( np.max( np.abs( data[ vec ] ) ) , 1e-4 )
        ret = [ data[ key ] for key in data.keys() ]
        return ret


    def stackImu( self, key, start ):
        idxs = np.arange( start, start + self.imuSteps )# % len( self.imuDict[ key ] )
        window = np.array( self.imuDict[ key ] )[ idxs ]
        # [ t, f ]
        return window


    def getHomeState( self, key, start ):
        idxs = np.arange( start, start + self.imuSteps )# % len( self.imuDict[ key ] )
        window = np.array( self.homeDict[ key ] )[ idxs ]
        m = np.mean( window, axis=0 )
        # [ t, f ]
        return m


    def stackFlow( self, video, start ):
        flowList = list()
        for i in range( start,
                        start + self.flowSteps * self.framePeriod,
                        self.framePeriod ):
            if self.nFlowMaps == 2:
                u, v = self.provideFlowFrame( video, i )
                flowList.append( np.array( [ u , v ] ) )
            elif self.nFlowMaps == 3:
                u, v, w = self.provideFlowFrame( video, i )
                flowList.append( np.array( [ u , v , w ] ) )
        stack = np.array( flowList )
        # [ h, w, {2, 3}, t ]
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



