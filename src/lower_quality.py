import cv2
import os


def make_dir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )



def convert_video( inp_dir, out_dir ):
    print(inp_dir)
    u_inp = os.path.join( inp_dir, 'u' )
    v_inp = os.path.join( inp_dir, 'v' )
    u_out = os.path.join( out_dir, 'u' ) 
    v_out = os.path.join( out_dir, 'v' )
    make_dir( u_out )
    make_dir( v_out )
    for filename in os.listdir( u_inp ):
        u = cv2.imread( os.path.join( u_inp, filename ) )
        v = cv2.imread( os.path.join( v_inp, filename ) )
        cv2.imwrite( os.path.join( u_out, filename ), 
                     u, [ cv2.IMWRITE_JPEG_QUALITY, 50 ] )
        cv2.imwrite( os.path.join( v_out, filename ), 
                     v, [ cv2.IMWRITE_JPEG_QUALITY, 50 ] )



def process_video( input_dir , output_dir ):
    for class_folder in os.listdir( input_dir ):
        class_inp_dir = os.path.join( input_dir , class_folder )
        class_out_dir = os.path.join( output_dir, class_folder )

        make_dir( class_out_dir )
        video_folders = os.listdir( class_inp_dir )
        for video_folder in video_folders:
            video_inp_folder =  os.path.join( class_inp_dir, video_folder ) 
            video_out_folder =  os.path.join( class_out_dir, video_folder ) 
            if os.path.exists( video_out_folder ):
                continue
            os.mkdir( video_out_folder )

            print( 'Converting' , video_folder )
            convert_video( video_inp_folder, video_out_folder )
            print( 'Done' )


input_dir  = '/media/olorin/Documentos/caetano/datasets/UCF-101_flow'
output_dir = '/home/olorin/Documents/caetano/datasets/UCF-101_flow'
process_video( input_dir, output_dir )
