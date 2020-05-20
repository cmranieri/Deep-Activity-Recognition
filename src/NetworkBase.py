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
                  trainListPath = '../splits/ucf101/trainlist01.txt',
                  testListPath  = '../splits/ucf101/testlist01.txt',
                  tl            = False,
                  tlSuffix      = '',
                  normalize     = False,
                  useFlips      = True,
                  nFlowMaps     = 2,
                  **kwargs ):
        self.dim           = dim
        self.flowSteps     = flowSteps
        self.imuSteps      = imuSteps
        self.classes       = classes
        self.flowDataDir   = flowDataDir
        self.rgbDataDir    = rgbDataDir
        self.imuDataDir    = imuDataDir
        self.modelDir      = modelDir
        self.modelName     = modelName
        self.tlSuffix      = tlSuffix
        self.streams       = streams
        self.normalize     = normalize
        self.framePeriod   = framePeriod
        self.clipTh        = clipTh
        self.useFlips      = useFlips
        self.lblFilename   = lblFilename
        self.trainListPath = trainListPath
        self.testListPath  = testListPath
        self.nFlowMaps     = nFlowMaps

        self._resultsDir  = '../results'
        self._outputsPath = os.path.join( '../outputs', self.modelName + '.pickle' )

        self.model = self.loadModel( restoreModel , tl )
        self._step = 0


    def defineNetwork( self ):
        raise NotImplementedError( 'Please implement this method' )


    def _updateForTL( self, base_model ):
        #base_model.layers.pop()
        y = base_model.layers[-2].output
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
        return model


    def loadModel( self, restoreModel, tl ):
        print( 'Loading model...' )
        if not ( restoreModel or tl ):
            model = self.defineNetwork()
        else:
            model = load_model( os.path.join( self.modelDir,
                                              str(self.modelName) + '.h5' ) )
        if tl:
            modelName = self.modelName + self.tlSuffix
            model = self._updateForTL( model )
            self._saveModel()
        print( 'Model loaded!' )
        return model



    def _generateTrainDataProvider( self,
                                    batchSize,
                                    numThreads,
                                    maxsize ):
        return TrainDataProvider( flowDataDir = self.flowDataDir,
                                  rgbDataDir  = self.rgbDataDir,
                                  imuDataDir  = self.imuDataDir,
                                  namesFilePath = self.trainListPath,
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
                                  useFlips    = self.useFlips,
                                  clipTh      = self.clipTh,
                                  nFlowMaps   = self.nFlowMaps )

    def _generateTestDataProvider( self,
                                   maxsize,
                                   numSegments ):
        return TestDataProvider( flowDataDir  = self.flowDataDir,
                                 rgbDataDir   = self.rgbDataDir,
                                 imuDataDir   = self.imuDataDir,
                                 namesFilePath = self.testListPath,
                                 numSegments  = numSegments,
                                 maxsize      = maxsize,
                                 classes      = self.classes,
                                 dim          = self.dim,
                                 lblFilename  = self.lblFilename,
                                 flowSteps    = self.flowSteps,
                                 imuSteps     = self.imuSteps,
                                 streams      = self.streams,
                                 normalize    = self.normalize,
                                 framePeriod  = self.framePeriod,
                                 useFlips     = self.useFlips,
                                 clipTh       = self.clipTh,
                                 nFlowMaps   = self.nFlowMaps )



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
               stepsToEval       = 20000,
               evalPer           = True ):
        train_acc_list  = list()
        train_loss_list = list()

        print( 'Training...' )
        while self._step < steps:
            with self._generateTrainDataProvider( batchSize  = batchSize,
                                            numThreads = numThreads,
                                            maxsize    = maxsize ) as trainDataProvider:
                # train stepsToEval before saving and evaluating
                for eval_idx in range( stepsToEval ):
                    batchDict , labels = trainDataProvider.getBatch()
                    batch = self._prepareBatch( batchDict )
                    # train the selected batch
                    tr = self.model.train_on_batch( batch, labels )
                    batch_loss = tr[ 0 ]
                    batch_acc  = tr[ 1 ]
                    train_acc_list  += [ batch_acc ]
                    train_loss_list += [ batch_loss ]

                    #print( self._step, batch_acc, batch_loss )
                    # periodically shows train acc and loss on the batches
                    if not self._step % stepsToTrainError:
                        train_accuracy = np.mean( train_acc_list  )
                        train_loss     = np.mean( train_loss_list )
                        print( 'Step: %d | Train acc: %s | Loss: %s'%(
                               self._step, str(train_accuracy), str(train_loss) ) )
                        self._storeResult( 'train.txt', str(self._step) + ' ' +
                                                        str(train_accuracy) + ' ' +
                                                        str(train_loss) + '\n' )
                        train_acc_list  = list()
                        train_loss_list = list()

                    self._step += 1
            # save model
            self._saveModel()
            # evaluate
            if evalPer:
                print( 'STEP %d: TEST'%( self._step ) )
                self.evaluate()


    def evaluate( self,
                  maxsize      = 8,
                  numSegments  = 5,
                  storeTests   = False ):
        t = time.time()
        test_acc_list = list()
        outs          = list()
        preds_list    = list()
        labels_list   = list()
        print( 'Evaluating...' )
        batch_idx = 0
        with self._generateTestDataProvider( maxsize = maxsize,
                                             numSegments  = numSegments ) as testDataProvider:
            while True:
                if not batch_idx % 200:
                    print( 'Evaluating sample', batch_idx )
                # load batch and check end of data
                batchTuple = testDataProvider.getBatch()
                if batchTuple is None:
                    break
                # prepare batch
                batchDict , labels = batchTuple
                batch = self._prepareBatch( batchDict )
                # provide flips, if required
                if self.useFlips:
                    # provides flipped batch
                    flipBatchDict, _ = testDataProvider.getBatch()
                    flipBatch = self._prepareBatch( flipBatchDict )
                    # if batch contains two streams or more
                    if isinstance( batch, list ):
                        # for each stream
                        for j in range( len(batch) ):
                            print( batch[j].shape )
                            batch[j] = np.concatenate( ( batch[j], flipBatch[j] ), axis = 0 )
                    # if batch contains a sigle stream
                    else:
                        batch = np.concatenate( ( batch, flipBatch ), axis = 0 )
                # if one of the streams runs out of instances, finish loop
                end_loop = False
                if isinstance( batch, list ):
                    for inp_id in range( len( self.model.inputs ) ):
                        if self.model.inputs[inp_id].shape[1] != batch[inp_id].shape[1]:
                            end_loop = True
                else:
                    if self.model.input.shape[1] != batch.shape[1]:
                        end_loop = True
                if end_loop:
                    continue
                # predict the data of an entire video
                y_ = self.model.predict( batch )
                # mean scores of each sample
                mean = np.mean( np.array( y_ ), 0 )
                # check whether the prediction is correct
                # assumes the label is the same for all instances of the batch
                correct_prediction = np.equal( np.argmax( mean ),
                                               np.argmax( labels[0] ) )
                test_acc_list.append( correct_prediction )
                # append outputs to lists
                if storeTests:
                    preds_list.append( mean )
                    labels_list.append( labels[0] )
                batch_idx += 1
        
        test_accuracy = np.mean( test_acc_list )
        print( 'Time elapsed:', time.time() - t )
        print( 'Test accuracy:', test_accuracy )
        self._storeResult( 'test.txt', str(self._step) + ' ' +
                                      str( test_accuracy ) + '\n' )
        if storeTests:
            with open( self._outputsPath, 'wb' ) as f:
                pickle.dump( dict( { 'predictions': np.array( preds_list ),
                                     'labels'     : np.array( labels_list ) } ), f )
        return test_accuracy
