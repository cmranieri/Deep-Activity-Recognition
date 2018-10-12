import numpy as np
from TrainLoader import TrainLoader
from TestLoader  import TestLoader

import keras
#from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.models import load_model, Model

import os
import time


class Network:
    
    def __init__( self, restoreModel = False ):
        self.timesteps = 10
        self.dim = 224
        self.classes = 101
        self.modelPath = '/media/olorin/Documentos/caetano/ucf101/models'
        
        self.defineNetwork( restoreModel )
      
        self.step = 0
        self.rootPath = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
        self.lblFilename = '../classInd.txt'
        self.trainFilenames = np.load( '../splits/trainlist01.npy' )
        self.testFilenames  = np.load( '../splits/testlist01.npy'  )
        #self.testFilenames    = np.load( '../splits/trainlist011.npy'  )
        self.resultsPath = '../results'


    def defineNetwork( self, restoreModel ):
        if not restoreModel:
            input_tensor = Input( shape = ( self.dim, self.dim,
                                            2 * self.timesteps) )
            self.model = InceptionV3( input_tensor = input_tensor,
                                      weights = None,
                                      include_top = True,
                                      pooling = None,
                                      classes = self.classes )
            optimizer = SGD( lr = 1e-2, momentum = 0.9,
                             nesterov=True, decay = 1e-5 )
            self.model.compile( loss = 'categorical_crossentropy',
                                optimizer = optimizer,
                                metrics   = [ 'acc' ] ) 


        else:
            self.model = load_model( os.path.join( self.modelPath,
                                                   'model.h5' ) )
        

    def generateTrainLoader( self ):
        return TrainLoader( self.rootPath,
                            self.trainFilenames,
                            self.lblFilename,
                            batchSize = 32,
                            timesteps = self.timesteps,
                            numThreads = 4,
                            maxsize = 10 )

    def generateTestLoader( self ):
        return TestLoader( self.rootPath,
                           self.testFilenames,
                           self.lblFilename,
                           numSegments = 25,
                           timesteps = self.timesteps,
                           numThreads = 4,
                           maxsize = 5 )



    def storeResult( self, filename, data ):
        f = open( os.path.join( self.resultsPath, filename ), 'a' )
        f.write( data )
        f.close()



    def train( self ):
        train_acc_list  = list()
        train_loss_list = list()
        trainFlag = True

        while self.step < 100000:
            with self.generateTrainLoader() as trainLoader:
                # saves and evaluates every n steps 
                while self.step % 10000 or trainFlag:
                    trainFlag = False

                    batch , labels = trainLoader.getBatch()
                    # train the selected batch
                    tr = self.model.train_on_batch( batch,
                                                    labels )
                    batch_loss = tr[ 0 ]
                    batch_acc  = tr[ 1 ]
                    train_acc_list  += [ batch_acc ]
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

                    self.step += 1
            # save model
            print( 'Saving model...' )
            self.model.save( os.path.join( self.modelPath,
                                           'model.h5' ) )
            print( 'Model saved!' )

            # evalutate model
            print( 'STEP %d: TEST'%( self.step ) )
            self.evaluate()
            trainFlag = True



    def evaluate( self ):
        t = time.time()
        test_acc_list  = list()
        i = 0
        print( 'Evaluating...' )
        with self.generateTestLoader() as testLoader:
            while not testLoader.endOfData():
                if i % 200 == 0:
                    print( 'Evaluating video', i )

                testBatch , testLabels = testLoader.getBatch()
                y_ = self.model.predict_on_batch( testBatch )
                mean = np.mean( y_, 0 )
                correct_prediction = np.equal( np.argmax( mean ),
                                               np.argmax( testLabels ) )
                
                if correct_prediction: test_acc_list.append( 1.0 )
                else: test_acc_list.append( 0.0 )
                
                #tst = self.model.test_on_batch( testBatch , testLabels )
                #test_acc_list.append( tst[ 1 ] )
                i += 1
            
        test_accuracy = np.mean( test_acc_list  )
        print( 'Time elapsed:', time.time() - t )
        print( 'test accuracy:', test_accuracy )
        self.storeResult( 'test.txt', str(self.step) + ' ' +
                                      str( test_accuracy ) + '\n' )
        return test_accuracy
        



if __name__ == '__main__':
    os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'
    
    network = Network( False )
    network.train()
    #network.evaluate()
