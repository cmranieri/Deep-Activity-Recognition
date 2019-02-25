
for fold in range(10):
    fold_id = '{:02d}'.format(fold+1)
    trainfile = open( 'trainlist'+fold_id+'.txt', 'w' )
    testfile  = open( 'testlist' +fold_id+'.txt', 'w' )

    for act in range(10):
        act_str = '{:02d}'.format(act+1)
        for seq in range( 9 ):
            tr_seq_str = '{:02d}'.format( (seq + fold) % 10 + 1 )
            trainfile.write( 'act'+act_str + 'seq'+tr_seq_str + '\n' )

        if fold == 0:
            te_seq_str = '{:02d}'.format( 10 )
        else:
            te_seq_str = '{:02d}'.format( fold % 10 )
        testfile.write( 'act'+act_str + 'seq'+te_seq_str + '\n' )
