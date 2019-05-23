import numpy as np
import sys
from mido import MidiFile, MidiTrack, Message, MetaMessage

def decode_midi(filename, maxnotes = 0):

    mid_in = MidiFile(filename)
    notes = []
    for track in mid_in.tracks:
        sum_of_ticks = 0
        pool = []
        for msg in track:
            sum_of_ticks += msg.time
            if msg.type=='note_on':
                for p in pool:
                    if p[1]==msg.channel and p[2]==msg.note:
                        if sum_of_ticks-p[0]>0: notes.append([p[0], p[2], sum_of_ticks-p[0]])
                        pool.remove(p)
                        break
                else: pool.append([sum_of_ticks, msg.channel, msg.note])
            if msg.type=='note_off':
                for p in pool:
                    if p[1]==msg.channel and p[2]==msg.note:
                        if sum_of_ticks-p[0]>0: notes.append([p[0], p[2], sum_of_ticks-p[0]])
                        pool.remove(p)
                        break
        for p in pool:
            if sum_of_ticks-p[0]>0: notes.append([p[0], p[2], sum_of_ticks-p[0]])

    notes = np.array(notes)
    ticks = np.unique(notes[:,0])

    pack = []
    for idx in range(len(ticks)-1):
        notes_at_ticks = np.unique(notes[notes[:,0]==ticks[idx]], axis=0)
        chord = str([p for p in notes_at_ticks[-maxnotes:,1]])
        pack.append(chord)
    return pack

def encode_midi(filename, data, tempo_set=500000):

    mid_out = MidiFile()
    track = MidiTrack()
    mid_out.tracks.append(track)

    track.append(Message('program_change', program=46, time=0))
    track.append(MetaMessage('set_tempo', tempo=tempo_set, time=0))
    for pack in data:

        chord = eval(pack)
        delay = 120
        for pit in chord:
            track.append(Message('note_on', note=pit, velocity=64, time=0))
        track.append(Message('note_off', note=chord[0], velocity=64, time=delay))
        for pit in chord[1:]:
            track.append(Message('note_off', note=pit, velocity=64, time=0))

    mid_out.save(filename)
