import cv2
import numpy as np
import os
import time


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



def convert_video( video_name, u_dir, v_dir ):
 
    cap = cv2.VideoCapture( video_name + '.avi' )
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor( frame1 , cv2.COLOR_BGR2GRAY )

    ur_dict = dict()
    vr_dict = dict()
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

        flow[ ... , 0 ] -= np.mean( flow[ ... , 0 ] )
        flow[ ... , 1 ] -= np.mean( flow[ ... , 1 ] )
        u = flow[ ..., 0 ].copy()

        v = flow[ ..., 1 ].copy()
        u = cv2.normalize( u, u, 0, 255, cv2.NORM_MINMAX )
        v = cv2.normalize( v, v, 0, 255, cv2.NORM_MINMAX )

        u_range = [ np.min( flow[ ... , 0 ] ),
                    np.max( flow[ ... , 0 ] ) ]
        v_range = [ np.min( flow[ ... , 1 ] ),
                    np.max( flow[ ... , 1 ] ) ]

        frame_id = str( 1000 + count )
        ur_dict[ frame_id ] = u_range
        vr_dict[ frame_id ] = v_range
        #store_frame_flow( u, v, frame_id, u_dir, v_dir )


        # TEST #
        #ut = cv2.imread( os.path.join( u_dir, frame_id + '.jpg' ),
        #                 cv2.IMREAD_GRAYSCALE )
        #u1 = np.array( ut, dtype = 'float32' ).copy()
        #cv2.normalize( u1, u1, u_range[0], u_range[1], cv2.NORM_MINMAX )
        # END TEST #

        prvs = next
        count += 1

    np.save( os.path.join( u_dir, 'range.npy' ), ur_dict )
    np.save( os.path.join( v_dir, 'range.npy' ), vr_dict )
    cap.release()



def process_video( input_dir , output_dir , raw_filename ):
    path_dirs = raw_filename.split('/')

    class_inp_dir = os.path.join( input_dir  , path_dirs[0] )
    class_out_dir = os.path.join( output_dir , path_dirs[0] )
    filename_no_ext = path_dirs[1].split('.')[0]
    make_dir( class_out_dir )
    filepath = os.path.join( class_inp_dir, filename_no_ext )

    make_dir( os.path.join( class_out_dir, filename_no_ext ) )
    u_dir = os.path.join( class_out_dir, filename_no_ext, 'u' )
    v_dir = os.path.join( class_out_dir, filename_no_ext, 'v' )
    make_dir( u_dir )
    make_dir( v_dir )

    print( 'Converting' , filename )
    convert_video( filepath, u_dir, v_dir )
    print( 'Done' )




os.environ[ 'CUDA_VISIBLE_DEVICES' ] = '0'

input_dir  = '/media/olorin/Documentos/caetano/datasets/UCF-101'
output_dir = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
trainlist = np.load( '../splits/trainlist01.npy' )
testlist = np.load( '../splits/testlist01.npy' )

#for filename in trainlist:
#    t = time.time()
#    process_video( input_dir, output_dir, filename )
#    print( 'Time:', time.time() - t )
for filename in testlist:
    t = time.time()
    process_video( input_dir, output_dir, filename )
    print( 'Time:', time.time() - t )
    



