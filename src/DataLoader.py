import numpy as np
import os
import glob
import time
import cv2

class DataLoader:

    def __init__( self,
                  rootPath,
                  filenames,
                  lblFilename ,
                  batchSize = 10,
                  dim = 224,
                  timesteps = 10 ):
        self.dim       = dim
        self.rootPath  = rootPath
        self.timesteps = timesteps
        self.filenames = filenames
        self.length    = filenames.shape[ 0 ]
        
        self.setBatchSize( batchSize )
        self.shuffle()
        self.generateLabelsDict( lblFilename )



    def generateLabelsDict( self, filename ):
        self.labelsDict = dict()
        f = open( filename , 'r' )
        for line in f.readlines():
            self.labelsDict[ line.split()[ 1 ] ] = line.split()[ 0 ]



    def shuffle( self ):
        self.ids = np.arange( self.length )
        np.random.shuffle( self.ids )
        self.index = 0


    def setBatchSize( self , batchSize ):
        self.batchSize = batchSize


    def incIndex( self ):
        self.index = self.index + self.batchSize
        if self.index >= self.length:
            self.shuffle()


    def randomBatchPaths( self ):
        batchPaths = list()
        endIndex = self.index + self.batchSize
        if endIndex > self.length:
            endIndex = self.length
        for i in range( self.index , endIndex):
            batchPaths += [ self.filenames[ self.ids[ i ] ].split('.')[0] ]
        self.curBatchLen = len( batchPaths )
        self.incIndex()
        return batchPaths


    def randomCrop( self , inp ):
        dim = self.dim
        imgDimX = inp.shape[ 1 ]
        imgDimY = inp.shape[ 0 ]
        left    = np.random.randint( imgDimX - dim + 1 )
        top     = np.random.randint( imgDimY - dim + 1 )
        right   = left + dim
        bottom  = top  + dim
        crop    = inp[ top : bottom , left : right ]
        return crop


    def randomFlip( self , inp ):
        if np.random.random() > 0.5:
            inp = np.flip( inp , 1 )
        return inp


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



    def randomBatchFlow( self ):
        batchPaths = self.randomBatchPaths()
        batch  = list()
        labels = list()
        for batchPath in batchPaths:
            fullPath  = os.path.join( self.rootPath, batchPath )
            filenames = sorted( [ os.path.basename(x) for x in glob.glob( 
                                          os.path.join( fullPath, 'u', '*.jpg' ) ) ] )

            begin_id = np.random.randint( len( filenames ) - self.timesteps )

            flowList = list()
            ur_dict = self.loadRanges( os.path.join( fullPath, 'u' ) )
            vr_dict = self.loadRanges( os.path.join( fullPath, 'v' ) )
            for filename in filenames[ begin_id : begin_id + self.timesteps ]:
                #u = cv2.imread( os.path.join( fullPath, 'u', filename ),
                #                cv2.IMREAD_GRAYSCALE )
                #v = cv2.imread( os.path.join( fullPath, 'v', filename ),
                #                cv2.IMREAD_GRAYSCALE )
                u = self.loadFlow( os.path.join( fullPath, 'u' ), filename, ur_dict )
                v = self.loadFlow( os.path.join( fullPath, 'v' ), filename, vr_dict )
                
                flowList += [ np.array( [ u , v ] ) ]
            videoFlow = np.array( flowList )
            videoFlow = np.transpose( videoFlow , [ 2 , 3 , 1 , 0 ] )
            videoFlow = self.randomCrop( videoFlow )
            videoFlow = self.randomFlip( videoFlow )
            batch  += [ videoFlow ]

            className = batchPath.split('/')[ 0 ]
            label = np.zeros(( 101 ) , dtype = 'float32')
            label[ int( self.labelsDict[ className ] ) - 1 ] = 1.0
            labels += [ label ]

        batch = np.array( batch, dtype = 'float32' )
        batch = np.reshape( batch , [ self.curBatchLen , 
                                      self.dim * self.dim * 2 * self.timesteps] )
        labels = np.array( labels )
        return batch, labels


if __name__ == '__main__':
    rootPath    = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
    # rootPath    = '/media/olorin/Documentos/caetano/datasets/UCF-101_flow'
    # rootPath    = '/home/caetano/Documents/datasets/UCF-101_flow'
    filenames   = np.load( '../splits/trainlist011.npy' )
    lblFilename = '../classInd.txt'
    dataLoader = DataLoader( rootPath, filenames, lblFilename )
    # batch, labels =  dataLoader.randomBatchFlow()
    for i in range( 100 ):
        t = time.time()
        batch, labels =  dataLoader.randomBatchFlow()
        print( i , batch.shape , labels.shape )
        print( 'Total time:' , time.time() - t )







