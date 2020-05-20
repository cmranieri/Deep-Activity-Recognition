"""Provides optical flow from all RGB files within a dataset
"""

import cv2
import numpy as np
import os
import time
from PIL import Image
from io import BytesIO
import pickle

algorithm = 'tvl1'
outsize   = ( 240, 320 )

def make_dir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )


def store_video_flow( u_list, v_list,
                      ur_list, vr_list,
                      out_dir, video_name ):
    print(out_dir, video_name)
    with open( os.path.join( out_dir , video_name + '.pickle' ) , 'wb' ) as f :
        pickle.dump( { 'u' : np.array( u_list ),
                       'v' : np.array( v_list ),
                       'u_range' : np.array( ur_list ),
                       'v_range' : np.array( vr_list ) } , f )


def prep_flow_frame( u, v ):
    u = cv2.normalize( u, u, 0, 255, cv2.NORM_MINMAX )
    v = cv2.normalize( v, v, 0, 255, cv2.NORM_MINMAX )
    u = u.astype( 'uint8' )
    v = v.astype( 'uint8' )
    img_u = Image.fromarray( u )
    img_v = Image.fromarray( v )
    u_out = BytesIO()
    v_out = BytesIO()
    img_u.save( u_out , format='jpeg' , quality=90 )
    img_v.save( v_out , format='jpeg' , quality=90 )
    return u_out, v_out


def convert_video( video_path, out_dir, ext ):
    cap = cv2.VideoCapture( video_path + ext )
    ret, frame1 = cap.read()
    frame1 = cv2.resize( frame1, outsize, interpolation=cv2.INTER_AREA )
    prev = cv2.cvtColor( frame1 , cv2.COLOR_BGR2GRAY )
    if algorithm == 'brox':
        prev = np.float32( prev ) / 255.0

    u_list  = list()
    v_list  = list()
    ur_list = list()
    vr_list = list()

    count = 0
    optical_flow = cv2.DualTVL1OpticalFlow_create()
    while( ret ):
        ret, frame2 = cap.read()
        if not ret:
            break
        
        count += 1
        frame2 = cv2.resize( frame2, outsize, interpolation=cv2.INTER_AREA )
        next = cv2.cvtColor( frame2, cv2.COLOR_BGR2GRAY )

        if algorithm == 'brox':
            next = np.float32( next ) / 255.0
            flow = cv2.pythoncuda.gpuOpticalFlowBrox( prev, next, None )
        elif algorithm == 'tvl1':
            flow = optical_flow.calc( prev, next, None )
        elif algorithm == 'farneback':
            flow = cv2.pythoncuda.gpuOpticalFlowFarneback( prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0 )

        u_out, v_out = prep_flow_frame( flow[ ... , 0 ].copy(),
                                        flow[ ... , 1 ].copy() )
        u_list += [ u_out ]
        v_list += [ v_out ]
        u_range = [ np.min( flow[ ... , 0 ] ),
                    np.max( flow[ ... , 0 ] ) ]
        v_range = [ np.min( flow[ ... , 1 ] ),
                    np.max( flow[ ... , 1 ] ) ]
        ur_list += [ u_range ]
        vr_list += [ v_range ]
        prev = next

    video_name = video_path.split('/')[-1]
    store_video_flow( u_list, v_list, ur_list, vr_list, out_dir, video_name )
    cap.release()



def process_video( input_dir , output_dir , raw_filename ):
    path_dirs = raw_filename.split('/')

    class_inp_dir = os.path.join( input_dir  , path_dirs[0] )
    class_out_dir = os.path.join( output_dir , path_dirs[0] )
    filename_no_ext = path_dirs[1].split('.')[0]
    ext             = path_dirs[1].split('.')[1]
    make_dir( class_out_dir )
    filepath = os.path.join( class_inp_dir, filename_no_ext )

    print( 'Converting' , raw_filename )
    convert_video( filepath, class_out_dir, '.' + ext )
    print( 'Done' )


def read_fileslist( path ):
    namesList = list()
    with open( path, 'r' ) as f:
        for line in f:
            namesList.append( line.split(' ')[0].strip('\n') )
    return namesList


def process_ucf101():
    input_dir  = '/home/cmranieri/datasets/UCF-101'
    output_dir = '/home/cmranieri/datasets/UCF-101_flow'
    trainlist  = read_fileslist( '../splits/ucf101/trainlist01.txt' )
    testlist   = read_fileslist( '../splits/ucf101/testlist01.txt' )

    for filename in trainlist+testlist:
        t = time.time()
        process_video( input_dir, output_dir, filename )
        print( 'Time:', time.time() - t )
        

def process_multimodal():
    input_dir  = '/home/cmranieri/datasets/multimodal_dataset/video'
    output_dir = '/home/cmranieri/datasets/multimodal_dataset_flow'
    trainlist  = read_fileslist( '../splits/multimodal_datasetucf101/trainlist01.txt' )
    testlist   = read_fileslist( '../splits/multimodal_dataset/testlist01.txt' )

    for filename in trainlist+testlist:
        t = time.time()
        process_video( input_dir, output_dir, filename )
        print( 'Time:', time.time() - t )


def process_utd_mhad():
    input_dir  = '/home/cmranieri/datasets/UTD-MHAD/RGB'
    output_dir = '/home/cmranieri/datasets/UTD-MHAD/flow'
    trainlist  = read_fileslist( '../splits/utd/trainlist01.txt' )
    testlist   = read_fileslist( '../splits/utd/testlist01.txt' )

    for filename in trainlist+testlist:
        t = time.time()
        process_video( input_dir, output_dir, filename )
        print( 'Time:', time.time() - t )


def process_lyell():
    input_dir  = '/lustre/cranieri/datasets/lyell/rgb_blurred'
    output_dir = '/lustre/cranieri/datasets/lyell/flow'
    trainlist  = read_fileslist( '../splits/lyell/trainlist01.txt' )
    testlist   = read_fileslist( '../splits/lyell/testlist01.txt' )

    for filename in trainlist+testlist:
        t = time.time()
        process_video( input_dir, output_dir, filename )
        print( 'Time:', time.time() - t )


if __name__ == '__main__':
    #process_ucf101()
    #process_multimodal()
    #process_utd_mhad()
    #process_lyell()

    help(make_dir)
