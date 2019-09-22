# -*- coding: utf-8 -*-

import os
import re
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


    def load_data( self,
                   data_dir    = '',
                   window_size = None,
                   max_len     = None ):
        data_dict = dict()
        filenames = os.listdir( data_dir )
        for filename in filenames:
            if filename.split('.')[-1] != 'csv': continue
            classname = re.match('(\D+\d+).*', filename).groups()[0]
            with open( os.path.join(data_dir, filename), 'r' ) as f:
                inp = list( csv.reader(f, quoting=csv.QUOTE_NONNUMERIC) )
                if max_len is not None:
                    inp = self.fix_seq_size( inp, max_len )
                if window_size is not None:
                    inp = self.window_data( inp, window_size )
                data_dict[ classname + '/' + filename.split('.')[0] ] = inp
        return data_dict


if __name__ == '__main__':
    inertialLoader = InertialLoader()
    data_dir = '/home/cmranieri/datasets/UTD-MHAD/Inertial_csv'
    data = inertialLoader.load_data( data_dir )
    #print(np.array(data['act02/act02seq02']).shape)
