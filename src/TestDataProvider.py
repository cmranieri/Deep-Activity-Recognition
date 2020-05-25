import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock, Semaphore
import queue
from DataProvider import DataProvider


class TestDataProvider( DataProvider ):
    def __init__( self, streams, numSegments, **kwargs ):
        super( TestDataProvider , self ).__init__( **kwargs )
        self._numSegments  = numSegments
        self.streams       = streams
        self._batchesMutex = Lock()
        self._finishedSem  = Semaphore(0)
        self._paths        = self._getPaths()
        
 
    def _processedAll( self ):
        return self._totalProcessed >= self._length


    def endOfData( self ):
        return ( self._processedAll() and
                 self._batchQueue.empty() )

    
    def _reset( self ):
        self._index = 0
        self._totalProcessed = 0


    def _incIndex( self ):
        self._index += 1


    def _getLabelArray( self, path ):
        labels = np.zeros( ( 5 * self._numSegments, self.classes ), 
                           dtype = 'float32' )
        className = path.split('/')[ -2 ]
        labels[ :, int( self._labelsDict[ className ] ) - 1 ] = 1.0
        self._labels = labels
        return labels


    def generateFlowBatch( self, path ):
        batch = list()
        fullPath  = os.path.join( self.flowDataDir, path )
        video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
        for i in range( self._numSegments ):
            space = len( video[ 'u' ] ) // self._numSegments
            if i * space + self.flowSteps * self.framePeriod < len( video[ 'u' ] ):
                start = i * space
            else:
                start = len( video[ 'u' ] ) - 1 - self.flowSteps * self.framePeriod
            # [ b, h, w, {2,3}, t ]
            batch.append( self.stackFlow( video, start ) )
        batch = np.array( batch, dtype = 'float32' )
        batch = self.getCrops( inp = batch, stream = 'temporal' )
        return batch


    def generateRgbBatch( self, path ):
        batch = list()
        fullPath  = os.path.join( self.rgbDataDir, path )
        video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
        for i in range( self._numSegments ):
            space = len( video ) // self._numSegments
            batch.append( self.provideRgbFrame( video, i*space ) )
        batch = np.array( batch, dtype = 'float32' )
        batch = self.getCrops( inp = batch, stream = 'spatial' )
        return batch


    def generateImuBatch( self, path ):
        batch = list()
        fullPath = os.path.join( self.imuDataDir, path )
        key = path.split('.')[0]
        seq = self.imuDict[ key ]
        # for each segment of a trial
        for i in range( self._numSegments ):
            space = len( seq ) // self._numSegments
            if i * space + self.imuSteps < len( seq ):
                start = i * space
            else:
                start = len( seq ) - 1 - self.imuSteps
            # [ b, t, f ]
            batch.append( self.stackImu( key, start ) )
        batch = np.array( batch, dtype = 'float32' )
        batch = self.replicateImu( batch )
        return batch


    def generateBatch( self, path ):
        batchDict = dict()
        if 'temporal' in self.streams:
            batch = self.generateFlowBatch( path )
            batchDict['temporal'] = batch
        if 'spatial' in self.streams:
            batch = self.generateRgbBatch( path )
            batchDict['spatial'] = batch
        if 'inertial' in self.streams:
            batch = self.generateImuBatch( path )
            batchDict['inertial'] = batch
        labels = self._getLabelArray( path )
        return ( batchDict, labels )


    def _getPaths( self ):
        paths = list()
        for filename in self.filenames:
            name = filename.split('.')[0] 
            paths += [ name ]
        return paths


    def replicateImu( self, inp ):
        rep_inp = list( inp )
        if set( self.streams ).intersection( ['temporal', 'spatial'] ):
            # center + crops
            rep_inp = rep_inp * 5
        return np.array( rep_inp, dtype = 'float32' )


    def getCrops( self , inp, stream ):
        dim = self.dim
        marginX = ( inp.shape[ 2 ] - dim ) // 2
        marginY = ( inp.shape[ 1 ] - dim ) // 2

        crops =  list()
        crops += [ inp[ :,    0    : dim  ,   0    : dim ] ]
        crops += [ inp[ :,  -dim-1 : -1   ,   0    : dim ] ]
        crops += [ inp[ :,    0    : dim  , -dim-1 : -1  ] ]
        crops += [ inp[ :,  -dim-1 : -1   , -dim-1 : -1  ] ]
        crops += [ inp[ :, marginY : marginY + dim,
                           marginX : marginX + dim ] ]
        crops = np.array( crops, dtype = 'float32' )
        if stream == 'temporal':
            # [ c * b, h, w, {2,3}t ]
            crops = np.reshape( crops, [ 5 * self._numSegments,
                                         self.dim, self.dim,
                                         self.nFlowMaps * self.flowSteps ] )
        elif stream == 'spatial':
            # [ c * b, h, w, 3 ]
            crops = np.reshape( crops, [ 5 * self._numSegments,
                                         self.dim, self.dim, 3 ] )
        return crops


    def nextBatch( self ):
        if self._processedAll():
            return None
        path = self._paths[ self._index ]
        self._incIndex()
        batchDict, labels = self.generateBatch( path )
        return ( batchDict, labels )


    def getFlips( self , inp ):
        inp = np.flip( inp , 2 )
        return inp


    def getFlippedBatch( self, batchTuple ):
        batchDict, labels = batchTuple
        newBatchDict = dict()
        for stream in self.streams:
            if stream in [ 'temporal', 'spatial' ]:
                batch = batchDict[ stream ].copy()
                fbatch = self.getFlips( batch )
                newBatchDict[ stream ] = fbatch
            elif stream == 'inertial':
                newBatchDict[ stream ] = batchDict[ stream ].copy()
        return ( newBatchDict, labels )


    def _batchThread( self ):
        while not self._processedAll():
            self._batchesMutex.acquire()
            batchTuple1 = self.nextBatch()
            if batchTuple1 is not None:
                batchTuples = [ batchTuple1 ]
                if self.useFlips and \
                        set( self.streams ).intersection( [ 'temporal', 'spatial' ] ):
                    batchTuples.append( self.getFlippedBatch( batchTuple1 ) )
                    
                for batchTuple in batchTuples:
                    self._batchQueue.put( batchTuple )
                self._totalProcessed += 1
            self._batchesMutex.release()
        self._finishedSem.release()

    
    def toFiles( self, batch, prefix = '' ):
        for i, flow in enumerate(batch['temporal']):
            for j in range( self.flowSteps ):
                frame = flow[ ..., j ].copy()
                frame = cv2.normalize( frame, frame, 0, 255, cv2.NORM_MINMAX )
                frame = np.array(frame, dtype='uint8')
                cv2.imwrite( 'test/%s_%d_%d.jpeg' % (prefix,i,j), frame )


    def getBatch( self ):
        if self.endOfData():
            self._finishedSem.acquire()
            return None
        batchTuple = self._batchQueue.get()
        return batchTuple


if __name__ == '__main__':
    flowDataDir = '/home/cmranieri/datasets/UCF-101_flow'
    #imuDataDir  = '/home/cmranieri/datasets/multimodal_inertial'
    
    with TestDataProvider( flowDataDir  = flowDataDir,
                           #imuDataDir   = imuDataDir,
                           namesFilePath = '../splits/ucf101/testlist01.txt',
                           lblFilename  = '../classInd.txt',
                           streams      = ['temporal'],
                           flowSteps    = 10,
                           numSegments  = 5 ) as testDataProvider:
         # for i in range(40):
         i = 0
         while not testDataProvider.endOfData():
            t = time.time()
            batchDict, labels = testDataProvider.getBatch()
            print( i, batchDict['temporal'].shape )
            batchDict, labels = testDataProvider.getBatch()
            print( i, batchDict['temporal'].shape )
            #testDataProvider.toFiles( batch, '{:02d}'.format(i) + '_' )
            print( testDataProvider._totalProcessed, 'Total time:' , time.time() - t )
            i += 1







