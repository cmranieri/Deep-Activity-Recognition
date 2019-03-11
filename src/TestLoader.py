import numpy as np
import os
import time
import cv2
import pickle
from PIL import Image
from threading import Thread, Lock
import queue
import DataLoader

class TestLoader( DataLoader.DataLoader ):

    def __init__( self,
                  dataDir,
                  filenames,
                  lblFilename,
                  classes = 101,
                  dim = 224,
                  timesteps = 10,
                  numThreads = 1,
                  maxsize=10,
                  numSegments = 25,
                  stream = 'temporal',
                  smallBatches = 1 ):
        super( TestLoader , self ).__init__( dataDir,
                                             filenames,
                                             lblFilename,
                                             classes,
                                             dim,
                                             timesteps, 
                                             numThreads,
                                             maxsize,
                                             ranges = True )
        self._numSegments = numSegments
        self._stream = stream
        self._smallBatches = smallBatches
        self._videoPaths  = self._getVideoPaths()
        
 
    def _processedAll( self ):
        return self._index >= self._length


    def endOfData( self ):
        return ( self._processedAll() and
                 self._batchQueue.empty() )

    
    def _reset( self ):
        self._index = 0


    def _incIndex( self ):
        self._index += 1


    def stackFlow( self, video, start ):
        return super( TestLoader , self ).stackFlow( video, start )


    def getVideoBatch( self, path, flip = False ):
        batch = list()
        fullPath  = os.path.join( self.dataDir, path )
        video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )

        for i in range( self._numSegments):
            if self._stream == 'temporal':
                space = len( video[ 'u' ] ) // self._numSegments
                if i * space + self._timesteps < len( video[ 'u' ] ):
                    start = i * space
                else:
                    start = len( video[ 'u' ] ) - 1 - self._timesteps
                # [ b, u, v, 2, t ]
                batch.append( self.stackFlow( video, start ) )

            elif self._stream == 'spatial':
                space = len( video ) // self._numSegments
                frame = np.asarray( Image.open( video [ i * space ] ),
                                    dtype = 'float32' )
                frame = frame / 255.0
                batch.append( frame )
        
        batch = np.array( batch, dtype = 'float32' )
        batch = self.getCrops( batch )
        return batch


    def _getVideoPaths( self ):
        videoPaths = list()
        for filename in self.filenames:
            name = filename.split('.')[0] 
            videoPaths += [ os.path.join( self.dataDir, name ) ]
        return videoPaths


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
            # [ c * b, u, v, 2t ]
            crops = np.reshape( crops, [ 5 * self._numSegments,
                                         self.dim, self.dim,
                                         2 * self._timesteps ] )
        elif self._stream == 'spatial':
            # [ c * b, h, v, 3 ]
            crops = np.reshape( crops, [ 5 * self._numSegments,
                                         self.dim, self.dim, 3 ] )
        return crops


    def nextVideoBatch( self ):
        self._indexMutex.acquire()
        if self._processedAll():
            self._indexMutex.release()
            return None
        videoPath = self._videoPaths[ self._index ]
        self._incIndex()
        self._indexMutex.release()

        batch = self.getVideoBatch( videoPath )

        labels = np.zeros( ( 5 * self._numSegments, 101 ), dtype = 'float32' )
        className = videoPath.split('/')[ -2 ]
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
            batchTuple1 = self.nextVideoBatch()
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

    
    def toFiles( self, batch, prefix='' ):
        i = 0
        for instance in batch:
            for frame in instance.transpose( 2,0,1 ):
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
    filenames   = np.load( '../splits/trainlist011.npy' )
    lblFilename = '../classInd.txt'
    
    with TestLoader( dataDir, filenames, lblFilename, numThreads=2,
                     stream = 'temporal',
                     numSegments = 25, smallBatches = 5 ) as testLoader:
         for i in range(100):
         # while not testLoader.endOfData():
            t = time.time()
            batch, labels = testLoader.getBatch()
            #testLoader.toFiles( batch )
            # batch, labels =  testLoader.nextVideoBatch()
            # batch =  testLoader.getFlippedBatch( batch )
            print( testLoader._index, 'Total time:' , time.time() - t )







