import numpy as np
import os
import time

import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD

from TrainDataProvider import TrainDataProvider
from TestDataProvider  import TestDataProvider


class NetworkBase:
    def __init__( self,
                  modelDir,
                  modelName,
                  streams,
                  flowDataDir   = '',
                  rgbDataDir    = '',
                  imuDataDir    = '',
                  restoreModel  = False,
                  classes       = 101,
                  dim           = 224,
                  flowSteps     = 8,
                  imuSteps      = 20,
                  framePeriod   = 1,
                  clipTh        = 20,
                  lblFilename   = '../classInd.txt',
                  splitsDir     = '../splits/ucf101',
                  split_n       = '01',
                  tl            = False,
                  tlSuffix      = '',
                  normalize     = False ):
        self.dim         = dim
        self.flowSteps   = flowSteps
        self.imuSteps    = imuSteps
        self.classes     = classes
        self.flowDataDir = flowDataDir
        self.rgbDataDir  = rgbDataDir
        self.imuDataDir  = imuDataDir
        self.modelDir    = modelDir
        self.modelName   = modelName
        self.tlSuffix    = tlSuffix
        self.streams     = streams
        self.normalize   = normalize
        self.framePeriod = framePeriod
        self.clipTh      = clipTh
        
        self.lblFilename = lblFilename
        self._trainFilenames = np.load( os.path.join( splitsDir,
                                        'trainlist' + split_n + '.npy' ) )
        self._testFilenames  = np.load( os.path.join( splitsDir,
                                        'testlist'  + split_n + '.npy' ) )
        self._resultsDir = '../results'
        self._step = 0

        self.loadModel( restoreModel , tl )
        self._outputsPath = os.path.join( '../outputs', self.modelName + '.pickle' )


    def _defineNetwork( self ):
        raise NotImplementedError( 'Please implement this method' )


    def _changeTop( self ):
        base_model = self.model
        base_model.layers.pop()
        y = base_model.layers[-1].output
        y = Dense( self.classes, activation='softmax' )( y )
        model = Model( inputs = base_model.input , outputs = y )

        for layer in base_model.layers:
            layer.trainabe = False

        optimizer  = SGD( lr = 1e-3,
                          momentum = 0.9,
                          nesterov = True,
                          decay = 1e-5 )
        model.compile( loss = 'categorical_crossentropy',
                       optimizer = optimizer,
                       metrics = [ 'acc' ] )
        self.model = model


    def loadModel( self, restoreModel, tl ):
        print( 'Loading model...' )
        if not ( restoreModel or tl ):
            self.model = self._defineNetwork()
        else:
            self.model = load_model( os.path.join( self.modelDir,
                                                   str(self.modelName) + '.h5' ) )
        if tl:
            self.modelName = self.modelName + self.tlSuffix
            self._changeTop()
            self._saveModel()
        print( 'Model loaded!' )



    def _generateTrainDataProvider( self,
                                    batchSize,
                                    numThreads,
                                    maxsize ):
        return TrainDataProvider( flowDataDir = self.flowDataDir,
                                  rgbDataDir  = self.rgbDataDir,
                                  imuDataDir  = self.imuDataDir,
                                  filenames   = self._trainFilenames,
                                  batchSize   = batchSize,
                                  numThreads  = numThreads,
                                  maxsize     = maxsize,
                                  classes     = self.classes,
                                  dim         = self.dim,
                                  lblFilename = self.lblFilename,
                                  flowSteps   = self.flowSteps,
                                  imuSteps    = self.imuSteps,
                                  streams     = self.streams,
                                  normalize   = self.normalize,
                                  framePeriod = self.framePeriod,
                                  clipTh      = self.clipTh )

    def _generateTestDataProvider( self,
                                   maxsize,
                                   numSegments,
                                   smallBatches ):
        return TestDataProvider( flowDataDir  = self.flowDataDir,
                                 rgbDataDir   = self.rgbDataDir,
                                 imuDataDir   = self.imuDataDir,
                                 filenames    = self._testFilenames,
                                 numSegments  = numSegments,
                                 maxsize      = maxsize,
                                 smallBatches = smallBatches,
                                 classes      = self.classes,
                                 dim          = self.dim,
                                 lblFilename  = self.lblFilename,
                                 flowSteps    = self.flowSteps,
                                 imuSteps     = self.imuSteps,
                                 streams      = self.streams,
                                 normalize    = self.normalize,
                                 framePeriod  = self.framePeriod,
                                 clipTh       = self.clipTh )



    def _storeResult( self, filename, data ):
        f = open( os.path.join( self._resultsDir,
                                self.modelName + '_' + filename ), 'a' )
        f.write( data )
        f.close()


    def _prepareBatch( self, batch ):
        raise NotImplementedError( 'Please implement this method' )


    def _saveModel( self ):
        print( 'Saving model...' )
        self.model.save( os.path.join( self.modelDir,
                                       str(self.modelName) + '.h5' ) )
        print( 'Model saved!' )


    def train( self,
               steps,
               batchSize         = 16,
               numThreads        = 2,
               maxsize           = 6,
               stepsToTrainError = 100,
               stepsToEval       = 20000):
        train_acc_list  = list()
        train_loss_list = list()

        while self._step < steps:
            with self._generateTrainDataProvider( batchSize  = batchSize,
                                            numThreads = numThreads,
                                            maxsize    = maxsize ) as trainDataProvider:
                # train stepsToEval before saving and evaluating
                for i in range( stepsToEval ):
                    batch , labels = trainDataProvider.getBatch()
                    batch = self._prepareBatch( batch )
                    # train the selected batch
                    tr = self.model.train_on_batch( batch, labels )
                    batch_loss = tr[ 0 ]
                    batch_acc  = tr[ 1 ]
                    train_acc_list  += [ batch_acc ]
                    train_loss_list += [ batch_loss ]

                    # periodically shows train acc and loss on the batches
                    if not self._step % stepsToTrainError:
                        train_accuracy = np.mean( train_acc_list  )
                        train_loss     = np.mean( train_loss_list )
                        print( 'step %d, training accuracy %g, cross entropy %g'%(
                               self._step, train_accuracy, train_loss ) )
                        self._storeResult( 'train.txt', str(self._step) + ' ' +
                                                        str(train_accuracy) + ' ' +
                                                        str(train_loss) + '\n' )
                        train_acc_list  = list()
                        train_loss_list = list()

                    self._step += 1
            # save and evaluate model
            self._saveModel()
            print( 'STEP %d: TEST'%( self._step ) )
            self.evaluate()


    def evaluate( self,
                  maxsize      = 8,
                  numSegments  = 5,
                  smallBatches = 1,
                  storeTests   = False ):
        t = time.time()
        test_acc_list  = list()
        video_outs     = list()
        preds_list     = list()
        labels_list    = list()
        i = 0
        print( 'Evaluating...' )
        with self._generateTestDataProvider( maxsize      = maxsize,
                                       numSegments  = numSegments,
                                       smallBatches = smallBatches ) as testDataProvider:
            while not testDataProvider.endOfData():
                if not i % 200:
                    print( 'Evaluating video', i )

                testBatch , testLabels = testDataProvider.getBatch()
                testBatch = self._prepareBatch( testBatch )

                y_ = self.model.predict_on_batch( testBatch )
                # mean scores of each video
                video_outs += list( y_ )
               
                if not (i+1) % (smallBatches * 2):
                    mean = np.mean( np.array( video_outs ), 0 )
                    video_outs = list()
                    # check whether the prediction is correct
                    # assumes the label is the same for all small batches
                    correct_prediction = np.equal( np.argmax( mean ),
                                                   np.argmax( testLabels[0] ) )
                    test_acc_list.append( correct_prediction )

                    # store outputs
                    if storeTests:
                        preds_list.append( mean )
                        labels_list.append( testLabels[0] )
                i += 1
            
        test_accuracy = np.mean( test_acc_list )
        print( 'Time elapsed:', time.time() - t )
        print( 'test accuracy:', test_accuracy )
        self._storeResult( 'test.txt', str(self._step) + ' ' +
                                      str( test_accuracy ) + '\n' )
        if storeTests:
            with open( self._outputsPath, 'wb' ) as f:
                pickle.dump( dict( { 'predictions': np.array( preds_list ),
                                     'labels'     : np.array( labels_list ) } ), f )
        return test_accuracy
        



if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
