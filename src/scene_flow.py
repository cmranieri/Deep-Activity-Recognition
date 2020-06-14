import os
import subprocess
import numpy as np
import re
import cv2
from PIL import Image
from io import BytesIO
import pickle


sf_impair_path = os.path.join( '/home', 'caetano', 'PD-Flow', 'build', 'Scene-Flow-Impair' )
fps_depth = 15
fps_rgb   = 15


def prep_flow_frame( flow_frame ):
    out_list = list()
    ranges_list = list()
    # [ u, v, w ]
    for vec in flow_frame:
        ranges_list.append( [ np.min( vec ), np.max(vec) ] )
        vec = cv2.normalize( vec, vec, 0, 255, cv2.NORM_MINMAX )
        vec = vec.astype( 'uint8' )
        img = Image.fromarray( vec )
        out = BytesIO()
        img.save( out , format='jpeg' , quality=90 )
        out_list.append( out )
    return out_list, ranges_list


def prep_flow_video( data ):
    u_list = list()
    v_list = list()
    w_list = list()
    u_ranges = list()
    v_ranges = list()
    w_ranges = list()
    for flow_frame in data:
        out_list, ranges_list = prep_flow_frame( flow_frame )
        u_list.append( out_list[ 0 ] )
        v_list.append( out_list[ 1 ] )
        w_list.append( out_list[ 2 ] )
        u_ranges.append( ranges_list[ 0 ] )
        v_ranges.append( ranges_list[ 1 ] )
        w_ranges.append( ranges_list[ 2 ] )
    flow_video = { 'u': np.array( u_list ),
                   'v': np.array( v_list ),
                   'w': np.array( w_list ),
                   'u_range': np.array( u_ranges ),
                   'v_range': np.array( v_ranges ),
                   'w_range': np.array( w_ranges ) }
    return flow_video


def load_depth( depth_dir, lbl, fname ):
    fname_noext = fname.split( '.' )[0]
    d_path = os.path.join( depth_dir, lbl, '%s.npy'%fname_noext )
    depth = np.load( d_path )
    return depth


def load_depth_pair( depth, frame_id ):
    d1 = depth[ frame_id - 1 ]
    d2 = depth[ frame_id ]
    return d1, d2


def load_rgb( rgb_dir, lbl, fname ):
    fname_noext = fname.split( '.' )[0]
    rgb_path = os.path.join( rgb_dir, lbl, '%s.avi'%fname_noext )
    rgb = list()
    cap = cv2.VideoCapture( rgb_path )
    ret, f = cap.read()
    while ret:
        f = cv2.cvtColor( f, cv2.COLOR_BGR2GRAY )
        f = cv2.resize( f, ( 320, 240 ), interpolation=cv2.INTER_AREA )
        rgb.append( f )
        ret, f = cap.read()
    cap.release()
    return np.array( rgb )


def load_rgb_pair( rgb, frame_id ):
    if rgb.shape[ 0 ] <= frame_id:
        return None
    f1 = rgb[ frame_id-1 ]
    f2 = rgb[ frame_id ]
    return f1, f2


def get_results_fname():
    best_NN     = 0
    fname_best  = ''
    # Get filename of the last result
    for fname in os.listdir( '.tmp' ):
        res = re.match( 'pdflow_results(\d+).txt', fname )
        if res:
            NN = int( res.groups()[ 0 ] )
            if NN > best_NN:
                best_NN = NN
                fname_best = fname
    return fname_best


def results2array():
    fname = get_results_fname()
    data = np.zeros( [ 3, 240, 320 ] )
    with open( os.path.join( '.tmp', fname ), 'r' ) as f:
        for line in f.readlines():
            line = np.array( line.split(' '), dtype=np.float32 )
            data[ 0, int( line[0] ), int( line[1] ) ] = line[ 2 ]
            data[ 1, int( line[0] ), int( line[1] ) ] = line[ 3 ]
            data[ 2, int( line[0] ), int( line[1] ) ] = line[ 4 ]
    return data


def compute_pair_flow( depth, rgb, frame_id ):
    ret = load_rgb_pair( rgb, frame_id )
    if ret is None:
        return None
    f1, f2 = ret
    d1, d2 = load_depth_pair( depth, frame_id )
    if not os.path.exists( '.tmp' ):
        os.mkdir( '.tmp' )
    os.chdir( '.tmp' )
    cv2.imwrite( 'i1.png', f1 )
    cv2.imwrite( 'i2.png', f2 )
    cv2.imwrite( 'z1.png', d1 )
    cv2.imwrite( 'z2.png', d2 )
    subprocess.call( [ sf_impair_path + ' --no-show' ], shell=True, executable='/bin/bash' )
    os.chdir( '..' )
    data = results2array()
    [ os.remove( os.path.join( '.tmp', x ) ) for x in os.listdir( '.tmp' ) ]
    return data


def compute_video( depth_dir, rgb_dir, lbl, fname ):
    depth = load_depth( depth_dir, lbl, fname )
    rgb   = load_rgb( rgb_dir, lbl, re.sub( 'depth.npy', 'color.avi', fname ) )
    data = list()
    for frame_id in range( depth.shape[ 0 ] ): 
        print( 'Frame %d' % frame_id )
        pair_flow = compute_pair_flow( depth, rgb, frame_id )
        if pair_flow is not None:
            data.append( pair_flow )
        else:
            break
    return data


def compute_all( depth_dir, rgb_dir, out_dir ):
    for lbl in os.listdir( depth_dir ):
        for fname in sorted( os.listdir( os.path.join( depth_dir, lbl ) ) ):
            print( lbl, fname )
            fname_noext = fname.split( '.' )[0]
            video_outs   = list()
            video_ranges = list()
            s = re.findall( '_s\d+', fname )[0]
            t = re.findall( '_t\d+', fname )[0]
            if os.path.exists( os.path.join( out_dir, lbl, '%s.pickle'%fname_noext ) ):
                continue

            data = compute_video( depth_dir, rgb_dir, lbl, fname )
            flow_video = prep_flow_video( data )

            if not os.path.exists( os.path.join( out_dir, lbl ) ):
                os.mkdir( os.path.join( out_dir, lbl ) )
            with open( os.path.join( out_dir, lbl, '%s.pickle'%fname_noext ), 'wb' ) as f:
                pickle.dump( flow_video, f )


if __name__=='__main__':
    depth_dir = os.path.join( '/media', 'caetano', 'Caetano', 'datasets', 'UTD-MHAD', 'Depth_npy' )
    rgb_dir   = os.path.join( '/media', 'caetano', 'Caetano', 'datasets', 'UTD-MHAD', 'RGB' )
    out_dir   = os.path.join( '/media', 'caetano', 'Caetano', 'datasets', 'UTD-MHAD', 'scene_flow' )
    compute_all( depth_dir, rgb_dir, out_dir )


