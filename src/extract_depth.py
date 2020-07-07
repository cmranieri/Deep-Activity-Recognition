import cv2
import numpy as np
import os
import time
from PIL import Image
from io import BytesIO
import pickle
import re

outsize   = (320,240)

def make_dir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )


def read_fileslist( path ):
    namesList = list()
    with open( path, 'r' ) as f:
        for line in f:
            namesList.append( line.split(' ')[0].strip('\n') )
    return namesList


def store_video_rgb( frames_list,
                     out_dir, video_name ):
    print(out_dir, video_name)
    with open( os.path.join( out_dir , video_name + '.pickle' ) , 'wb' ) as f :
        pickle.dump( frames_list, f )


def prep_jpeg_frame( frame ):
    frame = cv2.resize( frame, outsize, interpolation=cv2.INTER_AREA )
    #frame = frame[ ... , [ 2 , 1 , 0 ] ]
    img_frame = Image.fromarray( frame )
    out = BytesIO()
    img_frame.save( out , format='jpeg' , quality=100 )
    return out


def prep_depth_frame( frame ):
    frame = cv2.resize( frame, outsize, interpolation=cv2.INTER_AREA )
    frame = ( frame * 5 ) / 256
    frame = frame.astype( 'uint8' )
    img_frame = Image.fromarray( frame )
    img_frame = img_frame.convert( 'RGB' )
    out = BytesIO()
    img_frame.save( out , format='jpeg' , quality=100 )
    return out


def convert_video( video_path, out_dir, ext ):
    depth_list = list()
    video = np.load( '%s.npy' % video_path )
    for frame in video:
        depth_list.append( prep_depth_frame( frame ) )

    #cap = cv2.VideoCapture( video_path + ext )

    #ret, frame = cap.read()
    #while ret:
    #    rgb_list.append( prep_jpeg_frame( frame ) )
    #    ret, frame = cap.read()

    video_name = video_path.split('/')[-1]
    store_video_rgb( depth_list, out_dir, video_name )
    #cap.release()



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



def process_utd_mhad():
    input_dir  = '/media/caetano/Caetano/datasets/UTD-MHAD/Depth_npy'
    output_dir = '/home/caetano/datasets/UTD-MHAD/depth_pickle'
    trainlist = read_fileslist( '../splits/utd-mhad/trainlist01.txt' )
    testlist = read_fileslist( '../splits/utd-mhad/testlist01.txt' )
    for filename in trainlist+testlist:
        filename = re.sub( 'color', 'depth', filename )
        t = time.time()
        process_video( input_dir, output_dir, filename )
        print( 'Time:', time.time() - t )   


def process_lyell():
    input_dir  = '/media/caetano/Caetano/lyell_dataset/lyell/depth'
    output_dir = '/home/caetano/datasets/lyell/depth_pickle'
    trainlist = read_fileslist( '../splits/lyell/trainlist01.txt' )
    testlist = read_fileslist( '../splits/lyell/testlist01.txt' )
    for filename in trainlist+testlist:
        t = time.time()
        process_video( input_dir, output_dir, filename )
        print( 'Time:', time.time() - t )   


if __name__ == '__main__':
    #process_ucf101()
    #process_multimodal()
    process_utd_mhad()
    #process_lyell()


