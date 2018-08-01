import numpy as np
import DataLoader
import TestLoader
import Temporal 
import os
import time


class Trainer:
    
    def __init__( self ):
        timesteps = 16
        self.network = Temporal.Temporal( timesteps = timesteps,
                                          modelsPath = '/lustre/cranieri/models/ucf101/',
                                          restoreModel = True )
        
        self.timesteps   = timesteps
        self.rootPath    = '/lustre/cranieri/UCF-101_flow'
        self.lblFilename = '../classInd.txt'
        self.trainFilenames   = np.load( '../splits/trainlist01.npy' )
        # self.testFilenames    = np.load( '../splits/testlist01.npy'  )
        self.testFilenames    = np.load( '../splits/trainlist011.npy'  )
        self.resultsPath = '../results'


    def generateDataLoader( self ):
        return DataLoader.DataLoader( self.rootPath,
                                      self.trainFilenames,
                                      self.lblFilename,
                                      batchSize = 256,
                                      timesteps = self.timesteps,
                                      numThreads = 10,
                                      maxsize = 32 )

    def generateTestLoader( self ):
        return TestLoader.TestLoader( self.rootPath,
                                      self.testFilenames,
                                      self.lblFilename,
                                      numSegments = 5,
                                      timesteps = self.timesteps,
                                      numThreads = 8,
                                      maxsize = 32 )



    def storeResult( self, filename, data ):
        f = open( os.path.join( self.resultsPath, filename ), 'a' )
        f.write( data )
        f.close()



    def train( self ):
        network = self.network
        train_acc_list  = list()
        train_loss_list = list()
        trainFlag = True

        self.step = network.getGlobalStep().eval()
        while self.step < 80000:
            with self.generateDataLoader() as dataLoader:
               while self.step % 10000 or trainFlag:
                    trainFlag = False
                    #np.random.seed( self.step )

                    batch , labels = dataLoader.getBatch()
                    # train the selected batch
                    [_, batch_accuracy, batch_loss] = network.trainBatch( 
                                                      batch , labels,
                                                      dropout1 = 0.3 , dropout2 = 0.3,
                                                      learning_rate = 1e-2 )
                    train_acc_list  += [ batch_accuracy ]
                    train_loss_list += [ batch_loss ]

                    # periodically shows train acc and loss on the batches
                    if not self.step % 100:
                        train_accuracy = np.mean( train_acc_list  )
                        train_loss     = np.mean( train_loss_list )
                        print( 'step %d, training accuracy %g, cross entropy %g'%(
                               self.step, train_accuracy, train_loss ) )
                        self.storeResult( 'train.txt', str(self.step) + ' ' +
                                                       str(train_accuracy) + ' ' +
                                                       str(train_loss) + '\n' )
                        train_acc_list  = list()
                        train_loss_list = list()

                    self.step = network.getGlobalStep().eval()

            # save model
            network.saveModel()
            # evalutate model
            print( 'STEP %d: TEST'%( self.step ) )
            test_accuracy = self.evaluate()
            self.step = network.getGlobalStep().eval()
            trainFlag = True



    def evaluate( self ):
        t = time.time()
        network = self.network
        test_acc_list  = list()
        i = 0
        print( 'Evaluating...' )
        with self.generateTestLoader() as testLoader:
            while i<2:
                if True: #i % 200 == 0:
                    print( 'Evaluating video', i )
                testBatch , testLabels = testLoader.getBatch()
                y_ = network.evaluateActivs( testBatch, testLabels )
                print(y_)
                mean = np.mean( y_[0], 0 )
                correct_prediction = np.equal( np.argmax( mean ),
                                               np.argmax( testLabels[0] ) )
                if correct_prediction: test_acc_list.append( 1.0 )
                else: test_acc_list.append( 0.0 )
                i += 1

        test_accuracy = np.mean( test_acc_list  )
        print( 'Time elapsed:', time.time() - t )
        print( 'test accuracy:', test_accuracy )
        self.storeResult( 'test.txt', str(self.step) + ' ' +
                                      str( test_accuracy ) + '\n' )
        return test_accuracy
        



if __name__ == '__main__':
    # os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    trainer = Trainer()
    #trainer.train()
    trainer.evaluate()
