import os
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import cv2

def make_dir( path ):
    if not os.path.exists( path ):
        os.mkdir( path )

outsize   = (256,454)
inpdir    = '/home/cmranieri/datasets/multimodal_dataset_flow'
outdir    = '/home/cmranieri/datasets/multimodal_dataset_flow2'
trainlist = list(np.load( '../splits/multimodal_dataset/trainlist01.npy' ))
testlist  = list(np.load( '../splits/multimodal_dataset/testlist01.npy' ))

make_dir( outdir )
for filename in trainlist+testlist:
    filename = filename.split('.')[0]
    inp_path = os.path.join( inpdir, filename )
    print('Processing', inp_path)
    video = pickle.load( open( inp_path + '.pickle', 'rb' ) )
    out_path = os.path.join( outdir, filename )
    make_dir( '/'.join( out_path.split('/')[:-1] ) )

    u_list = list()
    v_list = list()
    ur_list = list()
    vr_list = list()
    for i in range( len(video['u']) ):
        u = np.asarray( Image.open( video ['u'][i] ), dtype = 'uint8' )
        v = np.asarray( Image.open( video ['v'][i] ), dtype = 'uint8' )
        u = cv2.resize( u, outsize, interpolation=cv2.INTER_AREA )
        v = cv2.resize( v, outsize, interpolation=cv2.INTER_AREA )
        img_u = Image.fromarray( u )
        img_v = Image.fromarray( v )
        u_out = BytesIO()
        v_out = BytesIO()
        img_u.save( u_out, format='jpeg', quality=100 )
        img_v.save( v_out, format='jpeg', quality=100 )

        u_list.append( u_out )
        v_list.append( v_out )
        ur_list.append( video['u_range'][i] )
        vr_list.append( video['v_range'][i] )
   
    with open( os.path.join( outdir , filename + '.pickle' ) , 'wb' ) as f :
        pickle.dump( { 'u' : np.array( u_list ),
                       'v' : np.array( v_list ),
                       'u_range' : np.array( ur_list ),
                       'v_range' : np.array( vr_list ) } , f )
    print('Processed', out_path)

