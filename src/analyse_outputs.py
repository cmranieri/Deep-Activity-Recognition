import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas



def get_path( dataset, stream, split ):
    return os.path.join( '..', 'outputs', 'model-%s-%s-%0.2d.pickle' % ( dataset, stream, split ) )


def compute_results( dataset,
                     n_splits,
                     w,
                     temp_flow='lstm',
                     temp_imu='lstm',
                     temp_bimod='lstm' ):
    acc_list = list()
    cf_list  = list()
    for j in range( n_splits ):
        n_rows = list()
        outs   = dict()

        for key in w.keys():
            if w[ key ] == 0:
                continue
            with open( get_path( dataset, key, j+1 ), 'rb' ) as f:
                outs[ key ] = pickle.load( f )
            outs[ key ][ 'predictions' ] = np.mean( outs[ key ][ 'predictions' ],
                                                    axis = 1 )
            n_rows.append( len( outs[ key ][ 'labels' ] ) )
            
        correct_list = list()
        preds   = list()
        labels  = list()
        # Test instances in a split
        for i in range( min( n_rows ) ):
            pred = list()
            # For each modality
            for key in outs.keys():
                pred.append( w[ key ] * outs[ key ][ 'predictions' ][ i ] )
                label = outs[ key ][ 'labels' ][ i ]
            pred = np.sum( pred, axis=0 )
            preds.append( np.argmax( pred ) )
            labels.append( np.argmax( label ) )
            correct = np.equal( np.argmax( pred ),
                                np.argmax( label ) )
            correct_list.append( correct )
        acc_list.append( np.mean( correct_list ) )
        cf = confusion_matrix( labels,
                               preds )
        cf_list.append( cf )
    return acc_list, cf_list



if __name__ == '__main__':
    dataset     = 'lyell'
    n_splits    = 8
    w = { 'imu_sh' : 1,
          'cnn-lstm'  : 0 }
    lbls = [ 'Cereals', 'Clean', 'Laptop', 'Newspaper', 'Sandwich', 'Smartphone', 'Table', 'Tea', 'Wash' ]

    acc_list, cf_list = compute_results( dataset, n_splits, w )
    mean_acc = np.mean( acc_list )
    std_acc  = np.std(  acc_list )
    sum_cf   = np.sum( cf_list, axis=0 )

    print( mean_acc, std_acc )

    df = pandas.DataFrame( sum_cf,
                           index = lbls,
                           columns = lbls )
    sn.set(font_scale=1.4)
    sn.heatmap( df, annot=True, annot_kws={"size": 12} )
    plt.tight_layout()
    plt.show()
