import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue
from LoaderBase import LoaderBase

class TrainLoader( LoaderBase ):

    def __init__( self,
                  batchSize = 16,
                  stream    = 'temporal',
                  **kwargs ): 
        super( TrainLoader , self ).__init__( **kwargs )
        self._batchSize  = batchSize
        self._stream     = stream
        self._indexMutex = Lock()
        self._resetMutex = Lock()
 

    def _reset( self ):
        self._ids = np.arange( self._length )
        np.random.shuffle( self._ids )
        self._index = 0
        return 0


    def _resetSafe( self ):
        self._resetMutex.acquire()
        self._resetUnsafe()
        self._resetMutex.release()
        return 0


    def getIndex( self ):
        self._resetMutex.acquire()
        index = self._index
        ids   = self._ids
        self._resetMutex.release()
        return index, ids


    def _incIndex( self ):
        self._resetMutex.acquire()
        newIndex = self._index + self._batchSize
        if newIndex + self._batchSize >= self._length:
            newIndex = self._reset()
        self._index = newIndex
        self._resetMutex.release()
        return newIndex


    def _popIndex( self ):
        self._indexMutex.acquire()
        index, ids = self.getIndex()
        self._incIndex()
        self._indexMutex.release()
        return index, ids


    def _selectBatchPaths( self ):
        index, ids = self._popIndex()
        batchPaths = list()
        endIndex = index + self._batchSize
        for i in range( index , endIndex ):
            batchPaths += [ self.filenames[ ids[ i ] ].split('.')[0] ]
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


    def _randomFlip( self , img ):
        if np.random.random() > 0.5:
            img = np.flip( img , 1 )
        return img


    def _getLabelArray( self, path ):
        className = path.split('/')[ 0 ]
        label = np.zeros(( self._classes ) , dtype = 'float32')
        label[ int( self._labelsDict[ className ] ) - 1 ] = 1.0
        return label


    def generateRgbBatch( self ):
        batchPaths = self._selectBatchPaths()
        batch  = list()
        labels = list()
        for batchPath in batchPaths:
            fullPath  = os.path.join( self.dataDir, batchPath )
            video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )

            frameId = np.random.randint( len( video ) )
            frame = self.loadRgb( video, frameId )
            frame = self._randomCrop( frame )
            frame = self._randomFlip( frame )

            batch.append( frame )
            labels.append( self._getLabelArray( batchPath ) )

        batch  = np.array( batch, dtype = 'float32' )
        labels = np.array( labels )
        return ( batch , labels )


    def stackFlow( self, video, start ):
        stack = super( TrainLoader, self ).stackFlow( video, start )
        stack = self._randomCrop( stack )
        stack = self._randomFlip( stack )
        return stack


    def generateFlowBatch( self ):
        batchPaths = self._selectBatchPaths()
        batch  = list()
        labels = list()

        for batchPath in batchPaths:
            fullPath  = os.path.join( self.dataDir, batchPath )
            video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )

            start = np.random.randint( len( video[ 'u' ] ) -
                        self._timesteps * self.framePeriod )
            batch.append( self.stackFlow( video, start ) )
            labels.append( self._getLabelArray( batchPath ) )

        batch = np.array( batch, dtype = 'float32' )
        batch = np.reshape( batch , [ len( batchPaths ), 
                                      self.dim,
                                      self.dim,
                                      2 * self._timesteps] )
        labels = np.array( labels, dtype='float32' )
        return ( batch , labels )



    def _batchThread( self ):
        while self._produce:
            if self._stream == 'temporal':
                batchTuple = self.generateFlowBatch()
            elif self._stream == 'spatial':
                batchTuple = self.generateRgbBatch()
            self._batchQueue.put( batchTuple )


    def getBatch( self ):
        batchTuple = self._batchQueue.get()
        return batchTuple


if __name__ == '__main__':
    #dataDir    = '/lustre/cranieri/UCF-101_flow'
    dataDir     = '/home/cmranieri/datasets/UCF-101_flow'
    filenames   = np.load( '../splits/ucf101/testlist01.npy' )
    lblFilename = '../classInd.txt'
    with TrainLoader( dataDir     = dataDir,
                      filenames   = filenames,
                      lblFilename = lblFilename,
                      numThreads  = 3,
                      stream = 'temporal' ) as trainLoader:
        for i in range( 100000 ):
            t = time.time()
            batch, labels =  trainLoader.getBatch()
            print( i , batch.shape , labels.shape )
            #for i, frame in enumerate(batch):
            #    cv2.imwrite(str(i)+'u.jpeg', frame[...,0])
            #    cv2.imwrite(str(i)+'v.jpeg', frame[...,1])

            print( 'Total time:' , time.time() - t )







