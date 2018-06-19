import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from io import BytesIO
from threading import Thread, Lock
import queue

class DataLoader:

    def __init__( self,
                  rootPath,
                  filenames,
                  lblFilename ,
                  batchSize = 20,
                  dim = 224,
                  timesteps = 16,
                  numThreads = 1 ):
        self.rootPath   = rootPath
        self.filenames  = filenames
        self.dim        = dim
        self.timesteps  = timesteps
        self.numThreads = numThreads
        self.length     = filenames.shape[ 0 ]
        
        self.setBatchSize( batchSize )
        self._shuffle()
        self._generateLabelsDict( lblFilename )

        self._produce = True
        self._batchQueue = queue.Queue( maxsize = 10 )
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



    def _shuffle( self ):
        self._ids = np.arange( self.length )
        np.random.shuffle( self._ids )
        self._index = 0


    def setBatchSize( self , batchSize ):
        self.batchSize = batchSize


    def _incIndex( self ):
        self._index += self.batchSize
        if self._index >= self.length:
            self._shuffle()


    def _randomBatchPaths( self ):
        self._indexMutex.acquire()
        if self._index + self.batchSize > self.length:
            self._incIndex()
        batchPaths = list()
        endIndex = self._index + self.batchSize
        for i in range( self._index , endIndex ):
            batchPaths += [ self.filenames[ self._ids[ i ] ].split('.')[0] ]
        self._incIndex()
        self._indexMutex.release()
        return batchPaths


    def _randomCrop( self , inp ):
        dim = self.dim
        imgDimX = inp.shape[ 1 ]
        imgDimY = inp.shape[ 0 ]
        left    = np.random.randint( imgDimX - dim + 1 )
        top     = np.random.randint( imgDimY - dim + 1 )
        right   = left + dim
        bottom  = top  + dim
        crop    = inp[ top : bottom , left : right ]
        return crop


    def _randomFlip( self , inp ):
        if np.random.random() > 0.5:
            inp = np.flip( inp , 1 )
        return inp


    def loadFlow( self, video, index ):
        u_img = Image.open( video ['u'] [index] )
        v_img = Image.open( video ['v'] [index] )
        u_range = video ['u_range'] [index]
        v_range = video ['v_range'] [index]

        u = np.array( u_img, dtype = 'float32' ).copy()
        v = np.array( v_img, dtype = 'float32' ).copy()
        cv2.normalize( u, u,  u_range[ 0 ], u_range[ 1 ], cv2.NORM_MINMAX )
        cv2.normalize( v, v,  v_range[ 0 ], v_range[ 1 ], cv2.NORM_MINMAX )
        return u, v


    def stackFlow( self, video, start, timesteps ):
        flowList = list()
        for i in range( start, start+timesteps ):
            u, v = self.loadFlow( video, i )
            flowList += [ np.array( [ u , v ] ) ]
        stack = np.array( flowList )
        stack = np.transpose( stack , [ 2 , 3 , 1 , 0 ] )
        stack = self._randomCrop( stack )
        stack = self._randomFlip( stack )
        return stack


    def randomBatchFlow( self ):
        batchPaths = self._randomBatchPaths()
        batch  = list()
        labels = list()
        for batchPath in batchPaths:
            fullPath  = os.path.join( self.rootPath, batchPath )
            video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )

            start = np.random.randint( len( video[ 'u' ] ) - self.timesteps )
            batch += [ self.stackFlow( video, start, self.timesteps ) ]

            className = batchPath.split('/')[ 0 ]
            label = np.zeros(( 101 ) , dtype = 'float32')
            label[ int( self.labelsDict[ className ] ) - 1 ] = 1.0
            labels += [ label ]

        batch = np.array( batch, dtype = 'float32' )
        batch = np.reshape( batch , [ len( batchPaths ), 
                                      self.dim * self.dim * 2 * self.timesteps] )
        labels = np.array( labels )
        return ( batch , labels )


    def _startThreads( self ):
        for i in range( self.numThreads ):
            print( 'Initializing thread %d' % ( i ) )
            t = Thread( target = self._batchThread )
            self._threadsList += [ t ]
            t.start()


    def _batchThread( self ):
        while self._produce:
            batchTuple = self.randomBatchFlow()
            self._queueMutex.acquire()
            self._batchQueue.put( batchTuple )
            self._queueMutex.release()


    def getBatch( self ):
        return self._batchQueue.get()



if __name__ == '__main__':
    rootPath    = '/home/cranieri/datasets/UCF-101_flow'
    # rootPath    = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
    filenames   = np.load( '../splits/trainlist011.npy' )
    lblFilename = '../classInd.txt'
    with DataLoader( rootPath, filenames, lblFilename ) as dataLoader:
        for i in range( 100 ):
            t = time.time()
            batch, labels =  dataLoader.getBatch()
            print( i , batch.shape , labels.shape )
            print( 'Total time:' , time.time() - t )







