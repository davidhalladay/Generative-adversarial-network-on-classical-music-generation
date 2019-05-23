from phraser import *
import numpy as np
from pypianoroll import Multitrack, Track
import os

#data = decode_midi(os.path.join("raw_classical_music_data",'mz_545_1.mid'))
data = decode_midi('./mz_545_1.mid')
for idx, chord in enumerate(data):
    print('#%d: %s' % (idx,chord))

encode_midi('test.mid', data)

all_chords = sorted(set(data))
n_chords = len(all_chords)
chords_to_idx = dict((v, i) for i,v in enumerate(all_chords))
idx_to_chords = dict((i, v) for i,v in enumerate(all_chords))
print('Total # of chords:',n_chords)
for key in chords_to_idx:
    print(key,'==>',chords_to_idx[key])
