import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas


def get_path( dataset, stream, split ):
    return os.path.join( '..', 'outputs', 'model-%s-%s-%0.2d.pickle' % ( dataset, stream, split ) )


def get_sequences( outs, key, num_segments=25 ):
    #print(key, outs[key]['predictions'].shape)
    corr_preds_list = list()
    corr_lbl_list   = list()
    for sess_id in range( outs[key][ 'predictions' ].shape[0] ):
        sess_preds = list()
        # for each crop (data augmentation for videos)
        for i in range( 5 ):
            idxs = i * num_segments + np.arange( num_segments )
            sess_preds.append( outs[key][ 'predictions' ][ sess_id, idxs ] )
        preds_seq  = np.max( sess_preds, axis=0 )
        lbl_id     = outs[key][ 'labels' ][ sess_id ]
        lbl_id     = np.argmax( lbl_id )
        corr_preds = preds_seq[:, lbl_id]
        corr_preds_list.append( corr_preds )
        corr_lbl_list.append( lbl_id )
    return np.array(corr_preds_list), corr_lbl_list


def compute_all_seqs( dataset,
                      n_splits,
                      w,
                      n_classes,
                      num_segments ):
    corr_preds = np.zeros( [n_classes, num_segments] )
    for j in range( n_splits ):
        outs   = dict()
        corr_preds_fold = [ list() for i in range( n_classes ) ]
        for key in w.keys():
            if w[ key ] == 0:
                continue
            with open( get_path( dataset, key, j+1 ), 'rb' ) as f:
                outs[ key ] = pickle.load( f )
                corr_preds_key, corr_lbl_key = get_sequences( outs, key )
                corr_preds[ corr_lbl_key ] += w[key] * corr_preds_key
    return ( corr_preds / n_splits )


def compute_results( dataset,
                     n_splits,
                     w,
                     temp_flow='lstm',
                     temp_imu='lstm',
                     temp_bimod='lstm' ):
    acc_list = list()
    cf_list  = list()
    preds_seq_list = list()
    for j in range( n_splits ):
        n_rows = list()
        outs   = dict()

        for key in w.keys():
            if w[ key ] == 0:
                continue
            with open( get_path( dataset, key, j+1 ), 'rb' ) as f:
                outs[ key ] = pickle.load( f )
            outs[ key ][ 'mean_predictions' ] = np.mean( outs[ key ][ 'predictions' ],
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
                pred.append( w[ key ] * outs[ key ][ 'mean_predictions' ][ i ] )
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


def results_cf( dataset, n_splits, n_classes, num_segments, w, ax, lbls=None ):
    acc_list, cf_list = compute_results( dataset, n_splits, w )
    mean_acc = np.mean( acc_list )
    std_acc  = np.std(  acc_list )
    sum_cf   = np.sum( cf_list, axis=0 )
    print( w.keys(), mean_acc, std_acc )
    df = pandas.DataFrame( sum_cf,
                           index = lbls,
                           columns = lbls )
    sn.heatmap( df, ax=ax, annot=True, cbar=False, annot_kws={"size": 12} )
    return


def all_results_cf( dataset, n_splits, n_classes, num_segments, w_list, titles, lbls=None ):
    sn.set(font_scale=1.4)
    fig, axs = plt.subplots( nrows=2, ncols=3, figsize=(12,10) )
    for i in range( 6 ):
        results_cf( dataset, n_splits, n_classes, num_segments, w_list[i], axs[i//3,i%3], lbls )
        axs[ i//3, i%3 ].set_title( titles[i] )
    fig.tight_layout()
    plt.savefig( '../images/cf.pdf' )


def all_seqs( dataset, n_splits, n_classes, num_segments, w_list, titles, lbls=None ):
    sn.set(font_scale=1.4)
    fig, axs = plt.subplots( nrows=2, ncols=3, figsize=(12,10) )
    for i in range( 6 ):
        pred_seqs = compute_all_seqs( dataset, n_splits, w_list[i], n_classes, num_segments )
        df = pandas.DataFrame( pred_seqs, index=lbls )
        cbar = i in [2, 5]
        sn.heatmap( df, ax=axs[i//3,i%3], annot=False, cbar=cbar, cmap='coolwarm', annot_kws={"size": 12} )
        axs[ i//3, i%3 ].set_title( titles[i] )
        axs[ i//3, i%3 ].set_xticks( [] )
        axs[ i//3, i%3 ].set_xlabel( 'Segment' )
    fig.tight_layout()
    plt.savefig( '../images/seqs.pdf' )


if __name__ == '__main__':
    dataset = 'lyell'
    n_splits = 8
    n_classes = 9
    num_segments = 25
    # imulstm2, imu_sh, cnn-lstm, slstm
    w_list = [ { 'cnn-lstm' : 1 },
               { 'slstm'    : 1 },
               { 'imulstm2' : 1 },
               { 'imu_sh'   : 1 },
               { 'imulstm2' : 1./7, 'cnn-lstm'  : 6./7 },
               { 'imu_sh'   : 1./7, 'cnn-lstm'  : 6./7 } ]
    titles = [ 'Optical flow',
               'Scene flow', 
               'IMU', 
               'IMU + smart home',
               'IMU + opt. flow', 
               'IMU + smart home\n+ opt. flow' ]
    lbls = [ 'Cereals', 'Tidy', 'Laptop', 'Newspaper', 'Sandwich', 'Smartphone', 'Table', 'Tea', 'Dishes' ]

    all_results_cf( dataset, n_splits, n_classes, num_segments, w_list, titles, lbls )
    all_seqs( dataset, n_splits, n_classes, num_segments, w_list, titles, lbls )
