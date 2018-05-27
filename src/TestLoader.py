import numpy as np
import os
import glob
import time
import cv2

class TestLoader:

    def __init__( self,
                  rootPath,
                  filenames,
                  lblFilename,
                  numFrames = 25,
                  dim = 224,
                  timesteps = 10 ):
        self.rootPath  = rootPath
        self.timesteps = timesteps
        self.filenames = filenames
        self.length    = filenames.shape[ 0 ]
        self.numFrames = numFrames
        self.videoPaths = self.getVideoPaths()
        
        self.dim = dim
        self.reset()
        self.generateLabelsDict( lblFilename )



    def loadRanges( self, flowDir ):
        r_file = np.load( os.path.join( flowDir, 'range.npy' ) )
        r_dict = r_file[()]
        return r_dict



    def loadFlow( self, flowDir, filename, r_dict ):
        r = r_dict[ filename.split('.')[0] ]
        img = cv2.imread( os.path.join( flowDir, filename ),
                          cv2.IMREAD_GRAYSCALE )
        flow = np.array( img, dtype = 'float32' ).copy()
        cv2.normalize( flow, flow,  r[ 0 ], r[ 1 ], cv2.NORM_MINMAX )
        return flow


    def generateLabelsDict( self, filename ):
        self.labelsDict = dict()
        f = open( filename , 'r' )
        for line in f.readlines():
            self.labelsDict[ line.split()[ 1 ] ] = line.split()[ 0 ]



    def reset( self ):
        self.finished = False
        self.index = 0


    def incIndex( self ):
        self.index += 1
        if self.index >= self.length:
            self.finished = True


    def getVideoBatch( self , path, flip = False ):
        # t = time.time()
        uList = list()
        vList = list()
        frameNames  = sorted( glob.glob( os.path.join( path, 'u', '*.jpg' ) ) )
        ur_dict = self.loadRanges( os.path.join( path, 'u' ) )
        vr_dict = self.loadRanges( os.path.join( path, 'v' ) )
        for frameName in frameNames:
            frameName = os.path.basename( frameName )
            #uList += [ cv2.imread( os.path.join( path, 'u', frameName ),
            #                       cv2.IMREAD_GRAYSCALE ) ]
            #vList += [ cv2.imread( os.path.join( path, 'v', frameName ),
            #                       cv2.IMREAD_GRAYSCALE ) ]
            uList += [ self.loadFlow( os.path.join( path, 'u' ),
                                      frameName, ur_dict ) ]
            vList += [ self.loadFlow( os.path.join( path, 'v' ),
                                      frameName, vr_dict ) ]

        space = len( uList ) // self.numFrames
        flowList = list()
        for i in range( self.numFrames):
            begin = i * space
            end   = i * space + self.timesteps 
            if end >= len( uList ):
                begin = len( uList ) - 1 - self.timesteps
                end   = len( uList ) - 1
            u = uList[ begin : end ]
            v = vList[ begin : end ]
            flowList += [ np.array( [ u , v ] ) ]
        flowArray = np.array( flowList )
        batch = self.getCrops( flowArray )
        batch = np.transpose( batch, [ 0, 3, 4, 1, 2 ] )
        # print( time.time() - t )
        return batch



    def getVideoPaths( self ):
        videoPaths = list()
        for i in range( self.length ):
            name = self.filenames[ i ].split('.')[0] 
            videoPaths += [ os.path.join( self.rootPath, name ) ]
        return videoPaths


    def getCrops( self , inp ):
        dim = self.dim
        marginX = ( inp.shape[ 4 ] - dim ) // 2
        marginY = ( inp.shape[ 3 ] - dim ) // 2

        crops =  list()
        crops += [ inp[ ... ,   0    : dim  ,   0    : dim ] ]
        crops += [ inp[ ... , -dim-1 : -1   ,   0    : dim ] ]
        crops += [ inp[ ... ,   0    : dim  , -dim-1 : -1  ] ]
        crops += [ inp[ ... , -dim-1 : -1   , -dim-1 : -1  ] ]
        crops += [ inp[ ... , marginY : -marginY,
                              marginX : -marginX ] ]
        crops = np.array( crops, dtype = 'float32' )
        crops = np.reshape( crops, [ 5 * self.numFrames, 2, self.timesteps,
                                     self.dim, self.dim] )
        return crops



    def getFlips( self , inp ):
        inp = np.flip( inp , 2 )
        return inp



    def nextVideoBatch( self ):
        videoPath = self.videoPaths[ self.index ]

        batch = self.getVideoBatch( videoPath )
        self.raw_batch = batch

        labels = np.zeros( ( 5 * self.numFrames, 101 ), dtype = 'float32' )
        className = videoPath.split('/')[ -2 ]
        labels[ :, int( self.labelsDict[ className ] ) - 1 ] = 1.0
        self.labels = labels

        batch  = np.reshape( batch , [ 5 * self.numFrames , 
                                       self.dim * self.dim * 2 * self.timesteps] )
        self.incIndex()
        return batch, labels


    def getFlippedBatch( self ):
        batch = self.getFlips( self.raw_batch )
        batch  = np.reshape( batch , [ 5 * self.numFrames , 
                                       self.dim * self.dim * 2 * self.timesteps] )
        return batch, self.labels



if __name__ == '__main__':
    rootPath = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
    # rootPath = '/media/olorin/Documentos/caetano/datasets/UCF-101_flow'
    # rootPath = '/home/caetano/Documents/datasets/UCF-101_flow'
    #filenames   = np.load( '../splits/testlist01.npy' )
    filenames   = np.load( '../splits/trainlist011.npy' )
    lblFilename = '../classInd.txt'
    
    testLoader = TestLoader( rootPath, filenames, lblFilename )

    while not testLoader.finished:
        t = time.time()
        batch, labels =  testLoader.nextVideoBatch()
        batch, labels =  testLoader.getFlippedBatch()

        print( testLoader.index, 'Total time:' , time.time() - t )







