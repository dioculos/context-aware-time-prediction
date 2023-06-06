#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Library Imports
from __future__ import print_function, division
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers import LSTM, GRU, SimpleRNN #Adequating layers
from keras.layers import Input #Adequating layers
from keras.utils.data_utils import get_file
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization #Adequating layers
from collections import Counter
import numpy as np
import random
import sys
import os
import getopt
import copy
import csv
import io
import time
import ipywidgets
import traitlets
from datetime import datetime
from math import log
import pandas as pd
# from ATLSTM_layer import ATLSTM_layer


# In[ ]:


def load_data(eventlog, path, sep="|"):
    # return pd.read_csv('../../datasets/'+path+'/%s' % eventlog, sep=sep).values
    return pd.read_csv(os.path.join(os.path.dirname(__file__),'../../datasets/'+path+'/%s') % eventlog, sep=sep).values


# # In[ ]:


def get_divisor(timeseqs):
    return np.mean([item for sublist in timeseqs for item in sublist])

def create_model_folder(name, dirc):
    i = 1
    
    path_dir = "../../../results/output_files/models/"+dirc
    if os.path.isdir(path_dir) == False:
        try:
            os.mkdir(path_dir)
        except OSError:
            print ("Creation of the directory %s failed" % path_dir)
        else:
            print ("Successfully created the directory %s " % path_dir)
    
    for i in range(100):
        new_name = name + "_v" + str(i)
        path_name = path_dir + "/" + new_name
        if os.path.isdir(path_name) == False:
            try:
                os.mkdir(path_name)
            except OSError:
                continue
            else:
                print ("Successfully created the directory %s " % path_name)
                break
                
    return new_name
# In[ ]:


def main(argv = None):
    if argv is None:
        argv = sys.argv

    inputfile = ""
    directory = ""
    sep=""
    num_add_feats = 0
    
    try:
        opts, args = getopt.getopt(argv, "hi:d:s:n:")
    except getopt.GetoptError:
        print(os.path.basename(__file__),
              "-i <input_file> -d <directory> -s <separator> -n <num_add_feats>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(os.path.basename(__file__),
                  "-i <input_file> -d <directory> -s <separator> -n <num_add_feats>")
            sys.exit()
        elif opt == "-i":
            inputfile = arg
        elif opt == "-d":
            directory = arg
        elif opt == "-s":
            sep = arg
        elif opt == "-n":
            num_add_feats = int(arg)
    
    begin_time = datetime.now()
     
    #helper variables
    lines = [] #these are all the activity seq
#     timeseqs = [] #time sequences (differences between two events)
#     timeseqs2 = [] #time sequences (differences between the current and first)
    lastcase = ''
    line = ''
    firstLine = True
    lines = []
    timeseqs = []
    timeseqs2 = []
    timeseqs3 = []
    timeseqs4 = []
    add_feats = []
    times = []
    times2 = []
    times3 = []
    times4 = []
    add_feat = -1
    numlines = 0
    casestarttime = None
    lasteventtime = None
        
    ascii_offset = 161

    spamreader = load_data(inputfile, directory, sep)
    
    for row in spamreader:
        t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        if row[0]!=lastcase:
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            if not firstLine:
                lines.append(line)
                timeseqs.append(times)
                timeseqs2.append(times2)
                timeseqs3.append(times3)
                timeseqs4.append(times4)
                add_feats.append(list(add_feat))
            line = ''
            times = []
            times2 = []
            times3 = []
            times4 = []
            add_feat = row[3:]
            # add_feat = int(row[3])
            numlines+=1
        line+=chr(int(row[1])+ascii_offset)
        timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
        timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
        timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
        timediff3 = timesincemidnight.seconds #this leaves only time even occured after midnight
        timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday() #day of the week
        times.append(timediff)
        times2.append(timediff2)
        times3.append(timediff3)
        times4.append(timediff4)
        # add_feats.append(add_feat)
        lasteventtime = t
        firstLine = False

    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)
    timeseqs4.append(times4)
    add_feats.append(add_feat)
    numlines+=1

    divisor = get_divisor(timeseqs) #average time between events
    print('divisor: {}'.format(divisor))
    divisor2 = get_divisor(timeseqs2) #average time between current and first events
    print('divisor2: {}'.format(divisor2))

    elems_per_fold = int(round(numlines/3))
    fold1 = lines[:elems_per_fold]
    fold1_t = timeseqs[:elems_per_fold]
    fold1_t2 = timeseqs2[:elems_per_fold]
    fold1_t3 = timeseqs3[:elems_per_fold]
    fold1_t4 = timeseqs4[:elems_per_fold]
    fold1_ft = add_feats[:elems_per_fold]
    with open(os.path.join(os.path.dirname(__file__),'../../results/output_files/folds/fold1.csv'), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row, timeseq in zip(fold1, fold1_t):
            spamwriter.writerow([s +'#{}'.format(t) for s, t in zip(row, timeseq)])

    fold2 = lines[elems_per_fold:2*elems_per_fold]
    fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
    fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
    fold2_t3 = timeseqs3[elems_per_fold:2*elems_per_fold]
    fold2_t4 = timeseqs4[elems_per_fold:2*elems_per_fold]
    fold2_ft = add_feats[elems_per_fold:2*elems_per_fold]
    with open(os.path.join(os.path.dirname(__file__),'../../results/output_files/folds/fold2.csv'), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row, timeseq in zip(fold2, fold2_t):
            spamwriter.writerow([s +'#{}'.format(t) for s, t in zip(row, timeseq)])

    fold3 = lines[2*elems_per_fold:]
    fold3_t = timeseqs[2*elems_per_fold:]
    fold3_t2 = timeseqs2[2*elems_per_fold:]
    fold3_t3 = timeseqs3[2*elems_per_fold:]
    fold3_t4 = timeseqs4[2*elems_per_fold:]
    fold3_ft = add_feats[2*elems_per_fold:]
    with open(os.path.join(os.path.dirname(__file__),'../../results/output_files/folds/fold3.csv'), 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row, timeseq in zip(fold3, fold3_t):
            spamwriter.writerow([s +'#{}'.format(t) for s, t in zip(row, timeseq)])

    lines = fold1 + fold2
    lines_t = fold1_t + fold2_t
    lines_t2 = fold1_t2 + fold2_t2
    lines_t3 = fold1_t3 + fold2_t3
    lines_t4 = fold1_t4 + fold2_t4
    lines_ft = fold1_ft + fold2_ft

    step = 1
    sentences = []
    softness = 0
    next_chars = []
    lines = list(map(lambda x: x+'!',lines)) #put delimiter symbol
    maxlen = max(list(map(lambda x: len(x),lines))) #find maximum line size

    # next lines here to get all possible characters for events and annotate them with numbers
    chars = list(map(lambda x: set(x),lines))
    chars = list(set().union(*chars))
    chars.sort()
    target_chars = copy.copy(chars)
    chars.remove('!')
    print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
    target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
    print(indices_char)

    sentences_t = []
    sentences_t2 = []
    sentences_t3 = []
    sentences_t4 = []
    sentences_ft = []
    next_chars_t = []
    next_chars_t2 = []
    next_chars_t3 = []
    next_chars_t4 = []
    next_chars_ft = []
    for line, line_t, line_t2, line_t3, line_t4, line_ft in zip(lines, lines_t, lines_t2, lines_t3, lines_t4, lines_ft):
        for i in range(0, len(line), step):
            if i==0:
                continue

            #we add iteratively, first symbol of the line, then two first, three...

            sentences.append(line[0: i])
            sentences_t.append(line_t[0:i])
            sentences_t2.append(line_t2[0:i])
            sentences_t3.append(line_t3[0:i])
            sentences_t4.append(line_t4[0:i])
            sentences_ft.append(line_ft)
            next_chars.append(line[i])
            if i==len(line)-1: # special case to deal time of end character
                next_chars_t.append(0)
                next_chars_t2.append(0)
                next_chars_t3.append(0)
                next_chars_t4.append(0)
            else:
                next_chars_t.append(line_t[i])
                next_chars_t2.append(line_t2[i])
                next_chars_t3.append(line_t3[i])
                next_chars_t4.append(line_t4[i])
            next_chars_ft.append(line_ft)
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    num_features = len(chars)+5+num_add_feats+1
    print('num features: {}'.format(num_features))
    X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
    y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
    y_t = np.zeros((len(sentences)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        leftpad = maxlen-len(sentence)
        next_t = next_chars_t[i]
        sentence_t = sentences_t[i]
        sentence_t2 = sentences_t2[i]
        sentence_t3 = sentences_t3[i]
        sentence_t4 = sentences_t4[i]
        # sentence_ft = sentences_ft[i][0]
        # sentence_ft2 = sentences_ft[i][1]
        # sentence_ft3 = sentences_ft[i][2]
        for t, char in enumerate(sentence):
            multiset_abstraction = Counter(sentence[:t+1])
            for c in chars:
                if c==char: #this will encode present events to the right places
                    X[i, t+leftpad, char_indices[c]] = 1
            X[i, t+leftpad, len(chars)] = t+1
            X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
            X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
            X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/86400
            X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/7
            # X[i, t+leftpad, len(chars)+5] = next_chars_t[t]
            if num_add_feats > 0:
                for f in range(num_add_feats):
                    X[i, t+leftpad, len(chars)+f+5] = sentences_ft[i][f]
            # X[i, t+leftpad, len(chars)+6] = sentence_ft
            # X[i, t+leftpad, len(chars)+7] = sentence_ft2
            # X[i, t+leftpad, len(chars)+8] = sentence_ft3
        for c in target_chars:
            if c==next_chars[i]:
                y_a[i, target_char_indices[c]] = 1-softness
            else:
                y_a[i, target_char_indices[c]] = softness/(len(target_chars)-1)
        y_t[i] = next_t/divisor
        np.set_printoptions(threshold=sys.maxsize)

    # build the model: 
    print('Build model...')
    print(X.shape)
    
    main_input = Input(shape=(maxlen, num_features), name='main_input')
    # old code for use with Attention Gate LSTM's
    # l1 = ATLSTM_layer(128, return_sequences=True)(main_input) # the shared layer

    # l2_1 = ATLSTM_layer(128, return_sequences=False)(l1)

    # l2_2 = ATLSTM_layer(128, return_sequences=False)(l1)

    # d1 = keras.layers.Dropout(.2)(l1)
    # d2_1 = keras.layers.Dropout(.2)(l2_1)
    # d2_2 = keras.layers.Dropout(.2)(l2_2)

    # train a 2-layer LSTM with one shared layer

    l1 = LSTM(128, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    l2_1 = LSTM(128, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)
    l2_2 = LSTM(128, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
    b2_2 = BatchNormalization()(l2_2)

    act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    time_output = Dense(1, kernel_initializer='glorot_uniform', name='time_output')(b2_2)
    
    model = Model(inputs=[main_input], outputs=[act_output, time_output])
    
    model_folder = ""
    if num_add_feats == 0:
        model_folder = create_model_folder("model_nofeat",directory)
    else:
        model_folder = create_model_folder("model_"+str(num_add_feats)+"_feat", directory)
    
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    
    model.compile(loss={'act_output':'categorical_crossentropy', 'time_output':'mae'}, optimizer=opt)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=42)
    
    model_checkpoint = ModelCheckpoint("../../../results/output_files/models/"+directory+"/"+model_folder+'/model_{epoch:02d}-{val_loss:.2f}.h5',
                                   monitor='val_loss',
                                   verbose=0,
                                   save_best_only=True,
                                   save_weights_only=False,
                                   mode='auto')
                                   
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    
    model.fit({'main_input': X}, {'act_output':y_a, 'time_output':y_t},
          validation_split=0.2,
          verbose=2,
          callbacks=[early_stopping, model_checkpoint, lr_reducer],
          batch_size=maxlen,
          epochs=200
          )
    
    print(datetime.now() - begin_time)

if __name__ == "__main__":
    main(sys.argv[1:])
