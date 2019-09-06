# -*- coding: utf-8 -*-

import os
import numpy as np
import csv

class InertialLoader:

    def window_data( self, data, window_size ):
        w_data = [ data[ i*window_size : (i+1)*window_size ]
                   for i in range( len(data) // window_size ) ]
        return w_data


    def fix_seq_size( inp, max_len ):
        #Fill with zeros the sequences smaller than max_len
        if( len(inp) < max_len ):
            inp += np.zeros( [max_len-len(inp), 19] ).tolist()
        # Truncate sequences greater than max_len
        inp = inp[:max_len]
        return inp


    def load_multimodal( self,
                         data_dir    = '',
                         window_size = None,
                         max_len     = None):
        num_acts  = 20
        num_seqs  = 10
        data_dict = dict()

        for act_id in range( num_acts ):
            for seq_id in range( num_seqs ):
                act_num = '{:02d}'.format( act_id+1 )
                seq_num = '{:02d}'.format( seq_id+1 )
                filename = 'act' + act_num + 'seq' + seq_num + '.csv'
                with open( os.path.join(dataset_dir, filename), 'r' ) as f:
                    inp = list( csv.reader(f, quoting=csv.QUOTE_NONNUMERIC) )
                    if max_len is not None:
                        inp = self.fix_seq_size( inp, max_len )
                    if window_size is not None:
                        inp = self.window_data( inp, window_size )
                    data_dict[ 'act%s/act%sseq%s' % (act_num, act_num, seq_num) ] = inp
        return data_dict


if __name__ == '__main__':
    inertialLoader = InertialLoader()
    data_dir = '/home/olorin/Documents/caetano/datasets/multimodal_inertial'
    data = inertialLoader.load_multimodal( data_dir )
    print(np.array(data['act02/act02seq02']).shape)
