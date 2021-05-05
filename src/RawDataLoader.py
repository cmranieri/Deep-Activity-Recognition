# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import csv
from sklearn.preprocessing import Normalizer

class RawDataLoader:

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


    def convert_rate( self, data, input_rate, output_rate ):
        out   = list()
        ratio = output_rate / input_rate
        next_idx = 0.0
        for i, line in enumerate(data):
            print(i)
            if i == int( next_idx ):
                out.append( line )
                next_idx += ratio
        return out


    def normalize_all( self, data_dict ):
        all_data = list()
        for key in data_dict:
            all_data += list( data_dict[key] )
        normalizer = Normalizer()
        normalizer.fit( all_data )
        for key in data_dict:
            data_dict[key] = normalizer.transform( data_dict[key] )
        return data_dict


    def _read_classes( self, classInd ):
        classNames = list()
        with open( classInd, 'r' ) as f:
            for line in f.readlines():
                classNames.append( line.split(' ')[1].strip('\n') )
        return classNames


    def get_fnames( self, data_dir ):
        all_fnames = list()
        for root, dirs, fnames in os.walk( data_dir ):
            for fname in fnames:
                all_fnames.append( re.sub( data_dir+'/*', '', os.path.join( root, fname ) ) )
        return all_fnames


    def load_data( self,
                   data_dir    = '',
                   classInd    = '../classes/classIndLyell.txt',
                   window_size = None,
                   max_len     = None,
                   input_rate  = None,
                   output_rate = None,
                   diff_dirs   = True ):
        classNames = self._read_classes( classInd )
        data_dict = dict()

        #filenames = os.listdir( data_dir )
        filenames = self.get_fnames( data_dir )
        for filename in filenames:
            if filename.split('.')[-1] != 'csv': continue
            for classname in classNames:
                # CHANGE FOR UTD-MHAD
                if re.match( '%s_.*' % classname, filename ): break
            #classname = re.match('(\D+\d+).*', filename).groups()[0]
            with open( os.path.join(data_dir, filename), 'r' ) as f:
                inp = list( csv.reader(f, quoting=csv.QUOTE_NONNUMERIC) )
                if output_rate is not None:
                    inp = self.convert_rate( inp, input_rate, output_rate )
                if max_len is not None:
                    inp = self.fix_seq_size( inp, max_len )
                if window_size is not None:
                    inp = self.window_data( inp, window_size )
                if diff_dirs:
                    data_dict[ filename.split('.')[0] ] = inp
                else:
                    data_dict[ classname + '/' + filename.split('.')[0] ] = inp
        data_dict = self.normalize_all( data_dict )
        return data_dict
