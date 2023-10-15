#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
from keras.models import load_model
import csv
import copy
import sys
import getopt
import numpy as np
import distance
# from itertools import zip
from jellyfish._jellyfish import damerau_levenshtein_distance
# import unicodecsv
from sklearn import metrics
from math import sqrt
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd
#Begin explanations

import timeshap
from timeshap.utils import *
    
## Utilitary functions

begin_time = datetime.now()


# In[ ]:


def load_data(eventlog, path):
    #return pd.read_csv('../../../work/renato.alves/datasets/'+path+'/%s' % eventlog, sep="|", error_bad_lines=False).values
    return pd.read_csv(os.getcwd()+'/datasets/'+path+'/%s' % eventlog, sep="|")


# In[ ]:


def get_divisor(timeseqs):
    return np.mean([item for sublist in timeseqs for item in sublist])


# In[ ]:


def create_result_folder(name, dirc):
    i = 1
    
    path_dir = os.getcwd()+"/results/output_files/results/"+dirc
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


def encode(sentence, times, times3, feat, maxlen, num_add_feats, chars, char_indices, divisor, divisor2):
    num_features = len(chars)+5+num_add_feats+1
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    times2 = np.cumsum(times)
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t]-midnight
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        X[0, t+leftpad, len(chars)] = t+1
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = timesincemidnight.seconds/86400
        X[0, t+leftpad, len(chars)+4] = times3[t].weekday()/7
        if num_add_feats > 0:
            for f in range(num_add_feats):
                X[0, t+leftpad, len(chars)+f+5] = feat[f]
    return X


# In[ ]:


def getSymbol(predictions, target_indices_char):
    maxPrediction = 0
    symbol = ''
    i = 0;
    for prediction in predictions:
        if(prediction>=maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

# In[ ]:


def main(argv = None):
    
    if argv is None:
        argv = sys.argv

    inputfile = ""
    directory = ""
    model_path=""
    num_add_feats = 0
    
## Get parameters for execution

    try:
        opts, args = getopt.getopt(argv, "hi:d:m:n:")
    except getopt.GetoptError:
        print(os.path.basename(__file__),
              "-i <input_file> -d <directory> -m <model path> -n <num_features>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(os.path.basename(__file__),
                  "-i <input_file> -d <directory> -m <model path> -n <num_features>")
            sys.exit()
        elif opt == "-i":
            inputfile = arg
        elif opt == "-d":
            directory = arg
        elif opt == "-m":
            model_path = arg
        elif opt =="-n":
            num_add_feats = int(arg)
    
    begin_time = datetime.now()

    lastcase = ''
    line = ''
    firstLine = True
    lines = []
    caseids = []
    timeseqs = []  # relative time since previous event
    timeseqs2 = [] # relative time since case start
    timeseqs3 = [] # absolute time of previous event
    add_feats = []
    times = []
    times2 = []
    times3 = []
    add_feat = []
    numlines = 0
    casestarttime = None
    lasteventtime = None

    ascii_offset = 161

## Load all the data from the CSV 

    all_data = load_data(inputfile, directory)
    raw_model_features = list(all_data.columns)
    del raw_model_features[0:3]

    print (all_data.shape)
   
    spamreader = all_data.values
    
    for row in spamreader: ## Build up vectors containing the information in the csv + enhanced time statistics like time since midnight, etc 
        t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        if row[0]!=lastcase:
            caseids.append(row[0])
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            if not firstLine:        
                lines.append(line)
                timeseqs.append(times)
                timeseqs2.append(times2)
                timeseqs3.append(times3)
                add_feats.append(list(add_feat))
            line = ''
            times = []
            times2 = []
            times3 = []
            add_feat = row[3:]
            numlines+=1
        line+=chr(int(row[1])+ascii_offset)
        timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
        timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
        timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
        times.append(timediff)
        times2.append(timediff2)
        times3.append(datetime.fromtimestamp(time.mktime(t)))
        # feat = row[3]
        # servicetype.append(feat)
        lasteventtime = t
        firstLine = False
        
    # add last case
    lines.append(line)
    timeseqs.append(times)
    timeseqs2.append(times2)
    timeseqs3.append(times3)
    add_feats.append(add_feat)
    numlines+=1
    
    print ('numlines: {}'.format(numlines))

## Calculations for the enhanced time statistics

    divisor = get_divisor(timeseqs) #average time between events
    print('divisor: {}'.format(divisor))
    divisor2 = get_divisor(timeseqs2) #average time between current and first events
    print('divisor2: {}'.format(divisor2))
    divisor3 = np.mean(list(map(lambda x: np.mean(list(map(lambda y: x[len(x)-1]-y, x))), timeseqs2)))
    print('divisor3: {}'.format(divisor3))
    
    #%% Splits the data set into 3 folds, with the activitiy (line), caseID, timestamp and enhanced time statistics (timeseqs2)
    elems_per_fold = int(round(numlines/3))
    fold1 = lines[:elems_per_fold]
    fold1_c = caseids[:elems_per_fold]
    fold1_t = timeseqs[:elems_per_fold]
    fold1_t2 = timeseqs2[:elems_per_fold]

    fold2 = lines[elems_per_fold:2*elems_per_fold]
    fold2_c = caseids[elems_per_fold:2*elems_per_fold]
    fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
    fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]
    
    #%%
    fold3 = lines[2*elems_per_fold:]
    fold3_c = caseids[2*elems_per_fold:]
    fold3_t = timeseqs[2*elems_per_fold:]
    fold3_t2 = timeseqs2[2*elems_per_fold:]
    fold3_t3 = timeseqs3[2*elems_per_fold:]
    fold3_ft = add_feats[2*elems_per_fold:]

    ## 66% of the dataset being used here

    lines = fold1 + fold2
    caseids = fold1_c + fold2_c
    lines_t = fold1_t + fold2_t
    lines_t2 = fold1_t2 + fold2_t2

    ## Reformatting the activity representation
    
    step = 1
    sentences = []
    softness = 0
    next_chars = []
    lines = list(map(lambda x: x+'!',lines))
    maxlen = max(map(lambda x: len(x),lines))

    chars = list(map(lambda x : set(x),lines))
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

    

    ## determine our desired sequence length
    desired_length = maxlen

    # Find the maximum length of sequences
    max_seq_length = max(len(seq) for seq in lines)

    print (max_seq_length)

    # Pad the sequences to the maximum length on the left
    padded_lines = [seq.rjust(max_seq_length, '¥') for seq in lines]
    padded_lines_t = [[0] * (desired_length - len(seq)) + seq for seq in lines_t]
    padded_lines_t2 = [[0] * (desired_length - len(seq)) + seq for seq in lines_t2]

    # Create a Pandas DataFrame
    timeshap_data = {
        'CaseID': caseids,
        'Sequence': padded_lines,
        'TimeSinceLastEvent': padded_lines_t,
        'TimeSinceCaseStart': padded_lines_t2
    }

    timeshap_df = pd.DataFrame(timeshap_data)



    ## 33% of the dataset being used here

    lines = fold3
    caseids = fold3_c
    lines_t = fold3_t
    lines_t2 = fold3_t2
    lines_t3 = fold3_t3
    lines_ft = fold3_ft
    
    #set parameters, predicting for the size given by them
    predict_size = maxlen
    # load model, set this to the model generated by train.py
    model = load_model(os.getcwd()+'/results/output_files/models/'+directory+'/'+model_path)
    
    one_ahead_gt = []
    one_ahead_pred = []

    two_ahead_gt = []
    two_ahead_pred = []

    three_ahead_gt = []
    three_ahead_pred = []
    
    result_folder = ""
    if num_add_feats == 0:
        result_folder = create_result_folder("suffix_pred_nofeat",directory)
    else:
        result_folder = create_result_folder("suffix_pred_"+str(num_add_feats)+"_feats", directory)
    
    # make predictions            
    with open(os.getcwd()+'/results/output_files/results/'+directory+'/'+result_folder+'/suffix_and_remaining_time_%s' % inputfile, 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["CaseID", "Prefix length", "Ground truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times", "Predicted times", "RMSE", "MAE", "MAE_days"])
        ## After reading and writing some headers for the file, prediction starts, iteratively through each prefix size
        for prefix_size in range(2,maxlen):
            # print(prefix_size)
            print('Prevendo para prefixo: '+str(prefix_size)+'...')
            for line, caseid, times, times2, times3, ft in zip(lines, caseids, lines_t, lines_t2, lines_t3, lines_ft): 
                ## Pack the vector (actv, case, time, ets1, ets2, addf) and iterate it
                times.append(0)
                cropped_line = ''.join(line[:prefix_size])
                cropped_times = times[:prefix_size]
                cropped_times2 = times2[:prefix_size]
                cropped_times3 = times3[:prefix_size]
                if len(times2)<prefix_size:
                    continue # make no prediction for this case, since this case has ended already
                ground_truth = ''.join(line[prefix_size:prefix_size+predict_size])
                ground_truth_t = times2[prefix_size-1]
                case_end_time = times2[len(times2)-1]
                ground_truth_t = case_end_time-ground_truth_t
                predicted = ''
                total_predicted_time = 0
                for i in range(predict_size):
                    enc = encode(cropped_line, cropped_times, cropped_times3, ft, maxlen, num_add_feats, chars, char_indices, divisor, divisor2)
                    y = model.predict(enc, verbose=0) # make predictions
                    
                    ## Insert TimeShap

                    ## Validate the input on timeshap
                    padded_data = pd.read_csv(os.getcwd()+"\padded_event_log.csv", sep="|")
                    ids_for_test = np.random.choice(padded_data['CaseID'].unique(), size = 24, replace=False)
                    d_train = padded_data[~padded_data['CaseID'].isin(ids_for_test)]
                    d_train['CompleteTimestamp'] = pd.to_datetime(d_train['CompleteTimestamp'])
                    d_train['CompleteTimestamp'] = d_train['CompleteTimestamp'].apply(lambda x: int(x.timestamp()))
                    average_event = calc_avg_event(d_train, numerical_feats=['CompleteTimestamp'], categorical_feats=[])
                    print (average_event)
                    average_sequence = calc_avg_sequence(d_train, numerical_feats=['CompleteTimestamp'], categorical_feats=[], model_features=['CaseID','ActivityID','CompleteTimestamp'], entity_col=['CaseID'])
                    print (average_sequence)
                    avg_score_over_len = get_avg_score_with_avg_event(d_train, average_event, top=480)
                    return
                    #print(average_event)
                    # split predictions into seperate activity and time predictions
                    y_char = y[0][0] 
                    y_t = y[1][0][0]
                    prediction = getSymbol(y_char, target_indices_char) # undo one-hot encoding           
                    cropped_line += prediction
                    if y_t<0:
                        y_t=0
                    cropped_times.append(y_t)
                    if prediction == '!': # end of case was just predicted, therefore, stop predicting further into the future
                        one_ahead_pred.append(total_predicted_time)
                        one_ahead_gt.append(ground_truth_t)
                        # print('! predicted, end case')
                        break
                    y_t = y_t * divisor3
                    cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                    total_predicted_time = total_predicted_time + y_t
                    predicted += prediction
                output = []
                if len(ground_truth)>0:
                    output.append(caseid)
                    output.append(prefix_size)
                    output.append(ground_truth)
                    output.append(predicted)
                    output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                    dls = 1 - (damerau_levenshtein_distance(predicted, ground_truth) / max(len(predicted),len(ground_truth)))
                    if dls<0:
                        dls=0 # we encountered problems with Damerau-Levenshtein Similarity on some linux machines where the default character encoding of the operating system caused it to be negative, this should never be the case
                    output.append(dls)
                    output.append(1 - distance.jaccard(predicted, ground_truth))
                    output.append(ground_truth_t)
                    output.append(total_predicted_time)
                    output.append('')
                    output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                    #output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))
                    spamwriter.writerow(output)

    print(datetime.now() - begin_time)




# In[ ]:


if __name__ == "__main__":
    main(sys.argv[1:])

