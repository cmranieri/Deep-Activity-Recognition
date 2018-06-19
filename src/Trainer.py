import numpy as np
import DataLoader
import TestLoader
import Temporal 
import os
import time


class Trainer:
    
    def __init__( self ):
        timesteps = 10
        self.network = Temporal.Temporal( timesteps = timesteps,
                                          modelsPath = '/lustre/cranieri/models/ucf101/',
                                          restoreModel = False )
        
        self.timesteps   = timesteps
        self.rootPath    = '/home/cranieri/datasets/UCF-101_flow'
        self.lblFilename = '../classInd.txt'
        self.trainFilenames   = np.load( '../splits/trainlist01.npy' )
        self.testFilenames    = np.load( '../splits/testlist01.npy'  )
        self.resultsPath = '../results'
        self.testLoader  = TestLoader.TestLoader( self.rootPath,
                                                  self.testFilenames,
                                                  self.lblFilename,
                                                  numFrames = 5,
                                                  timesteps = self.timesteps )


    def generateDataLoader( self ):
        return DataLoader.DataLoader( self.rootPath,
                                      self.trainFilenames,
                                      self.lblFilename,
                                      batchSize = 128,
                                      timesteps = self.timesteps,
                                      numThreads = 10 )



    def storeResult( self, filename, data ):
        f = open( os.path.join( self.resultsPath, filename ), 'a' )
        f.write( data )
        f.close()



    def evaluate( self ):
        t = time.time()
        network = self.network
        test_acc_list  = list()
        i = 0
        print( 'Evaluating...' )
        while not self.testLoader.finished:
            if i % 200 == 0:
                print( 'Evaluating video', i )
            testBatch , testLabels = self.testLoader.nextVideoBatch()
            y_1 = network.evaluateActivs( testBatch, testLabels )
            testBatch , testLabels = self.testLoader.getFlippedBatch()
            y_2 = network.evaluateActivs( testBatch, testLabels )

            mean1 = np.mean( y_1[0], 0 )
            mean2 = np.mean( y_2[0], 0 )
            mean  = np.mean( [ mean1, mean2 ] , 0 )
            correct_prediction = np.equal( np.argmax( mean ),
                                           np.argmax( testLabels[0] ) )
            if correct_prediction: test_acc_list += [ 1.0 ]
            else: test_acc_list += [ 0.0 ]

            i += 1
        test_accuracy = np.mean( test_acc_list  )
        self.testLoader.reset()
        print( 'Time elapsed:', time.time() - t )
        return test_accuracy
        


    def train( self ):
        network = self.network
        train_acc_list  = list()
        train_loss_list = list()

        self.step = network.getGlobalStep().eval()
        with self.generateDataLoader() as dataLoader:
            while self.step < 60000:
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
                if self.step % 100 == 0:
                    train_accuracy = np.mean( train_acc_list  )
                    train_loss     = np.mean( train_loss_list )
                    print( 'step %d, training accuracy %g, cross entropy %g'%(
                           self.step, train_accuracy, train_loss ) )
                    self.storeResult( 'train.txt', str(self.step) + ' ' +
                                                   str(train_accuracy) + ' ' +
                                                   str(train_loss) + '\n' )
                    train_acc_list  = list()
                    train_loss_list = list()

                if self.step % 10000 == 0 and self.step != 0:
                    network.saveModel()
               
                if self.step % 10000 == 0 and self.step != 0:
                    print( 'STEP %d: TEST'%( self.step ) )
                    test_accuracy = self.evaluate()
                    print( 'test accuracy:', test_accuracy )
                    self.storeResult( 'test.txt', str(self.step) + ' ' +
                                                  str( test_accuracy ) + '\n' )

                self.step = network.getGlobalStep().eval()



if __name__ == '__main__':
    # os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '1'
    
    trainer = Trainer()
    trainer.train()
