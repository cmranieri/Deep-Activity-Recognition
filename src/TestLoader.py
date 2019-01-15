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
                  rootPath,
                  filenames,
                  lblFilename,
                  classes = 101,
                  dim = 224,
                  timesteps = 10,
                  numThreads = 1,
                  maxsize=10,
                  numSegments = 25,
                  tshape = False ):
        super( TestLoader , self ).__init__( rootPath,
                                             filenames,
                                             lblFilename,
                                             classes,
                                             dim, 
                                             timesteps, 
                                             numThreads,
                                             maxsize,
                                             ranges = True,
                                             tshape = tshape )
        self._numSegments = numSegments
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
        fullPath  = os.path.join( self.rootPath, path )
        video = pickle.load( open( fullPath + '.pickle' , 'rb' ) )
        space = len( video[ 'u' ] ) // self._numSegments

        for i in range( self._numSegments):
            if i * space + self._timesteps < len( video[ 'u' ] ):
                start = i * space
            else:
                start = len( video[ 'u' ] ) - 1 - self._timesteps
            # [ b, u, v, 2, t ]
            batch.append( self.stackFlow( video, start ) )
        batch = np.array( batch, dtype = 'float32' )
        batch = self.getCrops( batch )
        return batch


    def _getVideoPaths( self ):
        videoPaths = list()
        for filename in self.filenames:
            name = filename.split('.')[0] 
            videoPaths += [ os.path.join( self.rootPath, name ) ]
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
        # [ c * b, u, v, 2t ]
        crops = np.reshape( crops, [ 5 * self._numSegments,
                                     self.dim, self.dim,
                                     2 * self._timesteps ] )
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
            if batchTuple1 is not None:
                batchTuple2 = self.getFlippedBatch( batchTuple1 )
                self._batchQueue.put( batchTuple1 )
                self._batchQueue.put( batchTuple2 )
            else: break

    
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
    rootPath = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
    # rootPath = '/lustre/cranieri/UCF-101_flow'
    filenames   = np.load( '../splits/trainlist011.npy' )
    lblFilename = '../classInd.txt'
    
    with TestLoader( rootPath, filenames, lblFilename, numThreads=2 ) as testLoader:
         for i in range(100):
        # while not testLoader.endOfData():
            t = time.time()
            batch, labels = testLoader.getBatch()
            testLoader.toFiles( batch )
            # batch, labels =  testLoader.nextVideoBatch()
            # batch =  testLoader.getFlippedBatch( batch )
            print( testLoader._index, 'Total time:' , time.time() - t )







