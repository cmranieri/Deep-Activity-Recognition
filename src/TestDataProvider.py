import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue
from DataProvider import DataProvider

class TestDataProvider( DataProvider ):

    def __init__( self,
                  numSegments  = 25,
                  smallBatches = 1,
                  stream       = 'temporal',
                  **kwargs ):
        super( TestDataProvider , self ).__init__( **kwargs )
        self._numSegments  = numSegments
        self._stream       = stream
        self._smallBatches = smallBatches
        self._batchesMutex = Lock()
        self._paths = self._getPaths()
        
 
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


    #def stackFlow( self, video, start ):
    #    return super( TestDataProvider , self ).stackFlow( video, start )


    def generateFlowBatch( self, path, flip = False )
        batch = list()
        fullPath  = os.path.join( self.flowDataDir, path )
        video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
        for i in range( self._numSegments ):
            space = len( video[ 'u' ] ) // self._numSegments
            if i * space + self.flowSteps * self.framePeriod < len( video[ 'u' ] ):
                start = i * space
            else:
                start = len( video[ 'u' ] ) - 1 - self.flowSteps * self.framePeriod
            # [ b, h, w, 2, t ]
            batch.append( self.stackFlow( video, start ) )
        return batch


    def generateRgbBatch( self, path, flip = False ):
        batch = list()
        fullPath  = os.path.join( self.rgbDataDir, path )
        video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
        for i in range( self._numSegments ):
            space = len( video ) // self._numSegments
            batch.append( self.loadRgb( video, i*space ) )
        batch = np.array( batch, dtype = 'float32' )
        batch = self.getCrops( batch )
        return batch


    def generateImuBatch( self, path ):
        batch = list()
        fullPath = os.path.join( self.imuDataDir, path )
        key = path.split('.')[0]
        seq = self.imuDict[ key ]
        for i in range( self._numSegments ):
            space = len( seq ) // self._numSegments
            if i * space + self.imuSteps < len( seq ):
                start = i * space
            else:
                start = len( seq ) - 1 - self.imuSteps
            # [ b, t, f ]
            batch.append( self.stackImu( video, start ) )
        return batch


    def generateBatch( self, path, flip = False ):
        if self._stream == 'temporal':
            batch = generateFlowBatch( path, flip )
        elif self._stream == 'spatial':
            batch = generateRgbBatch( path, flip )
        return batch


    def _getPaths( self ):
        paths = list()
        for filename in self.filenames:
            name = filename.split('.')[0] 
            paths += [ name ]
        return paths


    def getCrops( self , inp ):
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
        if self._stream == 'temporal':
            # [ c * b, h, w, 2t ]
            crops = np.reshape( crops, [ 5 * self._numSegments,
                                         self.dim, self.dim,
                                         2 * self.flowSteps ] )
        elif self._stream == 'spatial':
            # [ c * b, h, w, 3 ]
            crops = np.reshape( crops, [ 5 * self._numSegments,
                                         self.dim, self.dim, 3 ] )
        return crops


    def nextBatch( self ):
        if self._processedAll():
            return None
        path = self._paths[ self._index ]
        self._incIndex()

        batch = self.generateBatch( path )

        labels = np.zeros( ( 5 * self._numSegments, self.classes ), 
                           dtype = 'float32' )
        className = path.split('/')[ -2 ]
        labels[ :, int( self._labelsDict[ className ] ) - 1 ] = 1.0
        self._labels = labels

        return ( batch , labels )


    def getFlips( self , inp ):
        inp = np.flip( inp , 2 )
        return inp


    def getFlippedBatch( self , batchTuple ):
        batch, labels = batchTuple
        fbatch = batch.copy()
        fbatch = self.getFlips( fbatch )
        return ( fbatch , labels )


    def _batchThread( self ):
        while True:
            self._batchesMutex.acquire()
            batchTuple1 = self.nextBatch()
            if batchTuple1 is None: break
            batchTuple2 = self.getFlippedBatch( batchTuple1 )

            batchStep = len( batchTuple1[0] ) // self._smallBatches
            for i in range( self._smallBatches ):
                sBatchTuple1 = ( batchTuple1[0][ i * batchStep : (i+1) * batchStep ],
                                 batchTuple1[1][ i * batchStep : (i+1) * batchStep ] )
                sBatchTuple2 = ( batchTuple2[0][ i * batchStep : (i+1) * batchStep ],
                                 batchTuple2[1][ i * batchStep : (i+1) * batchStep ] )
                self._batchQueue.put( sBatchTuple1 )
                self._batchQueue.put( sBatchTuple2 )
            self._totalProcessed += 1
            self._batchesMutex.release()

    
    def toFiles( self, batch, prefix='' ):
        i = 0
        for instance in batch:
                frame = instance * 255
            #for frame in instance.transpose( 2,0,1 ):
                frame = np.array(frame, dtype='uint8')
                cv2.imwrite( 'test/' + str(prefix) + str(i) + '.jpeg', frame )
                i += 1


    def getBatch( self ):
        if self.endOfData():
            return None
        batchTuple = self._batchQueue.get()
        return batchTuple


if __name__ == '__main__':
    dataDir = '/home/cmranieri/datasets/UCF-101_flow'
    # dataDir = '/lustre/cranieri/UCF-101_flow'
    filenames   = np.load( '../splits/ucf101/testlist01.npy' )
    lblFilename = '../classInd.txt'
    
    with TestDataProvider( dataDir = dataDir,
                           filenames = filenames,
                           lblFilename = lblFilename,
                           stream = 'temporal',
                           numSegments = 5,
                           smallBatches = 5 ) as testDataProvider:
         for i in range(100):
         # while not testDataProvider.endOfData():
            t = time.time()
            batch, labels = testDataProvider.getBatch()
            #testDataProvider.toFiles( batch, '{:02d}'.format(i) + '_' )
            print( testDataProvider._totalProcessed, 'Total time:' , time.time() - t )







