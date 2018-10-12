import cv2
import numpy as np
import os
import time
from PIL import Image
from io import BytesIO
import pickle


def make_dir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )


def flow2bgr( raw_flow ):
    hsv = np.zeros(( raw_flow.shape[ 0 ], 
                     raw_flow.shape[ 1 ],
                     3 ) , dtype = np.uint8)
    hsv[ ... , 1 ] = 255

    mag, ang = cv2.cartToPolar( raw_flow[ ... , 0 ] , raw_flow[ ... , 1 ] )
    hsv[ ... , 0 ] = ang * 180 / np.pi / 2
    hsv[ ... , 2 ] = cv2.normalize( mag , None , 0 , 255 , cv2.NORM_MINMAX )
    bgr = cv2.cvtColor( hsv , cv2.COLOR_HSV2BGR )
    return bgr



def store_frame_flow( u, v, frame_id, u_dir, v_dir ):
    cv2.imwrite( os.path.join( u_dir, frame_id + '.jpg' ),
                 u,
                 [ cv2.IMWRITE_JPEG_QUALITY , 50 ] )
    cv2.imwrite( os.path.join( v_dir, frame_id + '.jpg' ),
                 v, 
                 [ cv2.IMWRITE_JPEG_QUALITY , 50 ] )


def store_video_flow( u_list, v_list,
                      out_dir, video_name ):
    print(out_dir, video_name)
    with open( os.path.join( out_dir , video_name + '.pickle' ) , 'wb' ) as f :
        pickle.dump( { 'u' : np.array( u_list ),
                       'v' : np.array( v_list ) },  f )


def prep_flow_frame( flow ):
    #u = cv2.convertScaleAbs( flow[ ... , 0 ], alpha = 10 )
    #v = cv2.convertScaleAbs( flow[ ... , 1 ], alpha = 10 )
    #flow = np.array( [ u , v ] , dtype = np.uint8 )
    #cv2.normalize( flow, flow, 0, 255, cv2.NORM_MINMAX )
    u = np.array(flow[ ... , 0 ])
    v = np.array(flow[ ... , 1 ])
    cv2.normalize( u, u, 0, 255, cv2.NORM_MINMAX )
    cv2.normalize( v, v, 0, 255, cv2.NORM_MINMAX )
    u = u.astype( 'uint8' )
    v = v.astype( 'uint8' )
    img_u = Image.fromarray( u )
    img_v = Image.fromarray( v )
    u_out = BytesIO()
    v_out = BytesIO()
    img_u.save( u_out , format='jpeg' , quality=90 )
    img_v.save( v_out , format='jpeg' , quality=90 )
    return u_out, v_out


def convert_video( video_path, out_dir ):
 
    cap = cv2.VideoCapture( video_path + '.avi' )
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor( frame1 , cv2.COLOR_BGR2GRAY )

    u_list  = list()
    v_list  = list()
    count = 0
    while( ret ):
        ret, frame2 = cap.read()
        if not ret:
            break
        
        next = cv2.cvtColor( frame2,cv2.COLOR_BGR2GRAY )
        flow = cv2.calcOpticalFlowFarneback( prev       = prvs,
                                             next       = next, 
                                             flow       = None, 
                                             pyr_scale  = 0.5,
                                             levels     = 3,
                                             winsize    = 15,
                                             iterations = 3,
                                             poly_n     = 7,
                                             poly_sigma = 1.5,
                                             flags      = 0 )

        u_out, v_out = prep_flow_frame( np.copy(flow) )
        u_list += [ u_out ]
        v_list += [ v_out ]

        prvs = next
        count += 1

    video_name = video_path.split('/')[-1]
    store_video_flow( u_list, v_list, out_dir, video_name )
    cap.release()



def process_video( input_dir , output_dir , raw_filename ):
    path_dirs = raw_filename.split('/')

    class_inp_dir = os.path.join( input_dir  , path_dirs[0] )
    class_out_dir = os.path.join( output_dir , path_dirs[0] )
    filename_no_ext = path_dirs[1].split('.')[0]
    make_dir( class_out_dir )
    filepath = os.path.join( class_inp_dir, filename_no_ext )

    print( 'Converting' , filename )
    convert_video( filepath, class_out_dir )
    print( 'Done' )

#os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

# input_dir  = '/lustre/cranieri/UCF-101'
# output_dir = '/lustre/cranieri/UCF-101_flow'
input_dir  = '/media/olorin/Documentos/caetano/datasets/UCF-101'
output_dir = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
trainlist = list(np.load( '../splits/trainlist01.npy' ))
testlist = list(np.load( '../splits/testlist01.npy' ))

for filename in testlist:
    t = time.time()
    process_video( input_dir, output_dir, filename )
    print( 'Time:', time.time() - t )
#for filename in testlist:
#    t = time.time()
#    process_video( input_dir, output_dir, filename )
#    print( 'Time:', time.time() - t )
    



