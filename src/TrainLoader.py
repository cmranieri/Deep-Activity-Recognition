import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue
from DataLoader import DataLoader

class TrainLoader( DataLoader ):

    def __init__( self,
                  rootPath,
                  filenames,
                  lblFilename ,
                  dim = 224,
                  timesteps = 10,
                  numThreads = 1,
                  maxsize = 10,
                  batchSize = 32 ):
        
        super( TrainLoader , self ).__init__( rootPath,
                                              filenames,
                                              lblFilename,
                                              dim,
                                              timesteps,
                                              numThreads,
                                              maxsize )
        self.setBatchSize( batchSize )
        self._flip = False
 

    def _reset( self ):
        self._ids = np.arange( self._length )
        np.random.shuffle( self._ids )
        self._index = 0


    def setBatchSize( self , batchSize ):
        self.batchSize = batchSize


    def _incIndex( self ):
        self._index += self.batchSize
        if self._index >= self._length:
            self._reset()


    def _randomBatchPaths( self ):
        if self._index + self.batchSize > self._length:
            self._incIndex()
        batchPaths = list()
        endIndex = self._index + self.batchSize
        for i in range( self._index , endIndex ):
            batchPaths += [ self.filenames[ self._ids[ i ] ].split('.')[0] ]
        self._incIndex()
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
        #if np.random.random() > 0.5:
        if self._flip:
            inp = np.flip( inp , 1 )
        self._flip = not self._flip
        return inp



    def stackFlow( self, video, start ):
        stack = super( TrainLoader, self ).stackFlow( video, start )
        stack = self._randomCrop( stack )
        stack = self._randomFlip( stack )
        return stack


    def randomBatchFlow( self ):
        self._indexMutex.acquire()
        batchPaths = self._randomBatchPaths()
        self._indexMutex.release()
        batch  = list()
        labels = list()
        for batchPath in batchPaths:
            fullPath  = os.path.join( self.rootPath, batchPath )
            video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )

            start = np.random.randint( len( video[ 'u' ] ) - self._timesteps )
            batch.append( self.stackFlow( video, start ) )

            className = batchPath.split('/')[ 0 ]
            label = np.zeros(( 101 ) , dtype = 'float32')
            label[ int( self._labelsDict[ className ] ) - 1 ] = 1.0
            labels.append( label )

        batch = np.array( batch, dtype = 'float32' )
        batch = np.reshape( batch , [ len( batchPaths ), 
                                      self.dim,
                                      self.dim,
                                      2 * self._timesteps] )
        labels = np.array( labels )
        return ( batch , labels )



    def _batchThread( self ):
        while self._produce:
            batchTuple = self.randomBatchFlow()
            self._batchQueue.put( batchTuple )


    def getBatch( self ):
        batchTuple = self._batchQueue.get()
        return batchTuple


if __name__ == '__main__':
    #rootPath    = '/lustre/cranieri/UCF-101_flow'
    rootPath    = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
    filenames   = np.load( '../splits/trainlist01.npy' )
    lblFilename = '../classInd.txt'
    with TrainLoader( rootPath, filenames, lblFilename, numThreads = 1 ) as trainLoader:
        for i in range( 1 ):
            t = time.time()
            batch, labels =  trainLoader.getBatch()
            print( i , batch.shape , labels.shape )

            #for i, frame in enumerate(batch):
            #    cv2.imwrite(str(i)+'u.jpeg', frame[...,0])
            #    cv2.imwrite(str(i)+'v.jpeg', frame[...,1])

            print( 'Total time:' , time.time() - t )







