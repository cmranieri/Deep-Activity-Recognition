import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue
from DataProvider import DataProvider

class TrainDataProvider( DataProvider ):

    def __init__( self, batchSize, streams, **kwargs ): 
        super( TrainDataProvider , self ).__init__( **kwargs )
        self.batchSize   = batchSize
        self.streams     = streams
        self._indexMutex = Lock()
        self._resetMutex = Lock()
 

    def _reset( self ):
        self._ids = np.arange( self._length )
        np.random.shuffle( self._ids )
        self._index = 0
        return 0


    def _resetSafe( self ):
        with self._resetMutex:
            self._resetUnsafe()
        return 0


    def getIndex( self ):
        with self._resetMutex:
            index = self._index
            ids   = self._ids
        return index, ids


    def _incIndex( self ):
        with self._resetMutex:
            newIndex = self._index + self.batchSize
            if newIndex + self.batchSize >= self._length:
                newIndex = self._reset()
            self._index = newIndex
        return newIndex


    def _popIndex( self ):
        with self._indexMutex:
            index, ids = self.getIndex()
            self._incIndex()
        return index, ids


    def _selectBatchPaths( self ):
        index, ids = self._popIndex()
        batchPaths = list()
        endIndex = index + self.batchSize
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
        if self.useFlips and np.random.random() > 0.5:
            img = np.flip( img , 1 )
        return img


    def _getLabelArray( self, batchPaths ):
        labels = list()
        for path in batchPaths:
            className = path.split('/')[ 0 ]
            label = np.zeros(( self.classes ) , dtype = 'float32')
            label[ int( self._labelsDict[ className ] ) - 1 ] = 1.0
            labels.append( label )
        labels = np.array( labels )
        return labels


    # Override
    def stackFlow( self, video, start ):
        stack = super( TrainDataProvider, self ).stackFlow( video, start )
        stack = self._randomCrop( stack )
        stack = self._randomFlip( stack )
        return stack


    def generateRgbBatch( self, batchPaths, startsList ):
        batch  = list()
        for i, batchPath in enumerate( batchPaths ):
            fullPath  = os.path.join( self.rgbDataDir, batchPath )
            video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
            frameId = int( startsList[i] * ( len( video ) -
                           self.flowSteps * self.framePeriod ) )
            frame = self.provideRgbFrame( video, frameId )
            frame = self._randomCrop( frame )
            frame = self._randomFlip( frame )
            batch.append( frame )
        batch  = np.array( batch, dtype = 'float32' )
        return batch

    
    def generateFlowBatch( self, batchPaths, startsList ):
        batch  = list()
        for i, batchPath in enumerate( batchPaths ):
            fullPath  = os.path.join( self.flowDataDir, batchPath )
            video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
            start = int( startsList[i] * ( len( video['u'] ) -
                         self.flowSteps * self.framePeriod ) )
            batch.append( self.stackFlow( video, start ) )
        batch = np.array( batch, dtype = 'float32' )
        batch = np.reshape( batch , [ len( batchPaths ), 
                                      self.dim,
                                      self.dim,
                                      2 * self.flowSteps] )
        return batch


    def generateImuBatch( self, batchPaths, startsList ):
        batch  = list()
        for i, batchPath in enumerate( batchPaths ):
            key = batchPath.split('.')[ 0 ]
            seq = self.imuDict[ key ]
            start = int( startsList[i] * ( len(seq) - self.imuSteps ) )
            batch.append( self.stackImu( key, start ) )
        batch  = np.array( batch,  dtype = 'float32' )
        return batch


    def _batchThread( self ):
        while self._produce:
            batchDict = dict()
            batchPaths = self._selectBatchPaths()
            startsList = np.random.random( self.batchSize )
            if 'temporal' in self.streams:
                batch = self.generateFlowBatch( batchPaths, startsList )
                batchDict[ 'temporal' ] = batch
            if 'spatial' in self.streams:
                batch = self.generateRgbBatch( batchPaths, startsList )
                batchDict[ 'spatial' ] = batch
            if 'inertial' in self.streams:
                batch = self.generateImuBatch( batchPaths, startsList )
                batchDict[ 'inertial' ] = batch
            labels = self._getLabelArray( batchPaths )
            batchTuple = ( batchDict, labels )
            self._batchQueue.put( batchTuple )


    def getBatch( self ):
        batchTuple = self._batchQueue.get()
        return batchTuple


if __name__ == '__main__':
    flowDataDir = '/home/cmranieri/datasets/multimodal_dataset_flow'
    imuDataDir  = '/home/cmranieri/datasets/multimodal_dataset_imu'
    filenames   = '../splits/multimodal_10/trainlist01.txt'
    lblFilename = '../classIndMulti.txt'
    with TrainDataProvider( flowDataDir = flowDataDir,
                            imuDataDir  = imuDataDir,
                            lblFilename = lblFilename,
                            namesFilePath = filenames,
                            batchSize   = 16,
                            flowSteps   = 1,
                            numThreads  = 2,
                            streams = [ 'temporal' ] ) as trainDataProvider:
        for i in range( 100000 ):
            t = time.time()
            batch, labels =  trainDataProvider.getBatch()
            print( i , batch['temporal'].shape , labels.shape )
            #for i, frame in enumerate(batch):
            #    cv2.imwrite(str(i)+'u.jpeg', frame[...,0])
            #    cv2.imwrite(str(i)+'v.jpeg', frame[...,1])

            print( 'Total time:' , time.time() - t )







