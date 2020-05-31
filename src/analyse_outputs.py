import numpy as np
import pickle
import os


dataset     = 'lyell'
n_splits    = 8
temp_flow   = 'lstm'
temp_imu    = 'lstm'
temp_bimod  = 'lstm'
w = { 'flow' : 0,
      'imu'  : 0,
      'spat' : 0,
      'bimod': 0,
      'cnn'  : 1 }

def get_path( dataset, stream, split ):
    return os.path.join( '..', 'outputs', 'model-%s-%s-%0.2d.pickle' % ( dataset, stream, split ) )

acc_list = list()
for j in range( n_splits ):
    n_rows = list()
    outs   = dict()
    if w[ 'flow' ]:
        with open( get_path( dataset, 'v%s'%temp_flow, j+1 ), 'rb' ) as f:
            outs[ 'flow' ] = pickle.load( f )
        n_rows.append( len ( outs[ 'flow' ][ 'labels' ] ) )
    if w[ 'imu' ]:
        with open( get_path( dataset, 'imu%s'%temp_imu, j+1 ), 'rb' ) as f:
            outs [ 'imu' ] = pickle.load( f )
        n_rows.append( len ( outs[ 'imu' ][ 'labels' ] ) )
    if w[ 'spat' ]:
        with open( get_path( dataset, 'spatial', j+1 ), 'rb' ) as f:
            outs[ 'spat' ] = pickle.load( f )
        n_rows.append( len ( outs[ 'spat' ][ 'labels' ] ) )
    if w[ 'bimod' ]:
        with open( get_path( dataset, 'c%s'%temp_bimod, j+1 ), 'rb' ) as f:
            outs[ 'bimod' ] = pickle.load( f )
        n_rows.append( len ( outs[ 'bimod' ][ 'labels' ] ) )
    if w[ 'cnn' ]:
        with open( get_path( dataset, 'cnn', j+1 ), 'rb' ) as f:
            outs[ 'cnn' ] = pickle.load( f )
        n_rows.append( len ( outs[ 'cnn' ][ 'labels' ] ) )

    correct_list = list()
    # Test instances in a split
    for i in range( min( n_rows ) ):
        pred = list()
        for key in outs.keys():
            pred.append( w[ key ] * outs[ key ][ 'predictions' ][ i ] )
            label = outs[ key ][ 'labels' ][ i ]
        pred = np.sum( pred, axis=0 )
        correct = np.equal( np.argmax( pred ),
                            np.argmax( label ) )
        correct_list.append( correct )
    acc_list.append( np.mean( correct_list ) )

print(np.mean(acc_list))
print(np.std(acc_list))
