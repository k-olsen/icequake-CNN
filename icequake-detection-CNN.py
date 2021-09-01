
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:41:19 2021

@author: Kira
"""

import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime


plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = 10,6

# Define function to find nearest value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def sign_changes(array):
    output = []
    array = np.asarray(array)
    for i in range(len(array)-1): 
        e1 = array[i]
        e2 = array[i+1]
        es = [e1, e2]
        if np.sign(e1) == np.sign(e2):
            continue
        elif np.sign(e1) != np.sign(e2):
            diff1 = np.abs(e1)
            diff2 = np.abs(e2)
            min_val = np.min([diff1, diff2])
            if min_val == diff1:
                j = 0
            elif min_val == diff2:
                j = 1
            idx_multi = np.where(array == es[j])
            if len(idx_multi[0]) > 1:
                idx = find_nearest(idx_multi[0], i) # requires find_nearest be defined
            else:
                idx = idx_multi[0][0]
        output.append(idx)
    return output

# Contents:
# 1) code to plot icequake-detection results using CNN vs. old detection technique
# 2) code to build CNN (files to run this code not included due to space constraints)
#%% Read in tidal-model data points
os.chdir('/Users/Kira/OneDrive/Data Science/My_DS_Projects/Icequake_CNN_post/')

station = 'DR14'
file1 = 'DR14_CATS2008b_output_Dec2014-Dec2016_10min.txt' 
DR14 = pd.read_csv(file1, sep = '\s', header = 0)

# Format time for station
cats_tides = DR14    
yr = [el[6:] for el in cats_tides.date]
mo = [el[0:2] for el in cats_tides.date]
day = [el[3:5] for el in cats_tides.date]
hr = [el[0:2] for el in cats_tides.time]
m = [el[3:5] for el in cats_tides.time]
sec = [el[6:8] for el in cats_tides.time]
cats_t = []
for el in range(len(cats_tides)):
    cats_t.append(datetime(int(yr[el]),int(mo[el]),int(day[el]),int(hr[el]), int(m[el]),int(sec[el])) )                       
DR14['t'] = cats_t #append new list of t onto dataframe
#%% Identify low tides

# figure out how long tidal cycles are: ~22 - 24 hrs 
# use this to make bounds of time to investigate
# within bounds, find max value
# then 24 hrs later, find next high

tides = DR14 

zero_crossings3 = sign_changes(tides.z) #using this - 7/18/2020
cycle_starts = zero_crossings3[::2] #every other value
n_phases = len(cycle_starts) - 2

phase_z_list_l2l = [] #amplitude (z[m]) of each tidal point. High point to high point
phase_t_list_l2l = [] #times of tidal points
phase_start_list = []
alpha_cycle_list = []

errors = []
for i in range(n_phases):
    first = min(tides.z[cycle_starts[i]:cycle_starts[i+1]])
    first_idx = np.where(tides.z == first)
    if len(first_idx[0]) >1:
        if i == 0:
            first_idx = first_idx[0][0]
        else:
            first_idx_temp = first_idx
            first_idx = find_nearest(first_idx[0], last_idx) # requires find_nearest be defined. This red dot is ok!
    else:
        first_idx = first_idx[0][0]
    
    last = min(tides.z[cycle_starts[i+1]:cycle_starts[i+2]])
    last_idx = np.where(tides.z == last)
    if len(last_idx[0]) > 1:
        last_idx_temp = find_nearest(last_idx[0], first_idx) # requires find_nearest be defined    
        if last_idx_temp - first_idx < 30: # to avoid 2 very close lines
            if last_idx[0][1] - first_idx > 30:
                last_idx = last_idx[0][1]
            elif last_idx[0][2] - first_idx > 30:
                last_idx = last_idx[0][2]
        else:
            last_idx = last_idx_temp
    else:
        last_idx = last_idx[0][0]
    if last_idx - first_idx < 30:
        errors.append([i, first_idx, first_idx_temp, last_idx, last_idx_temp])
    phase_z_list_l2l.append(tides.z[first_idx:last_idx])
    phase_t_list_l2l.append(tides.t[first_idx:last_idx])
    phase_start_list.append(first_idx)
    
print('errors: %s' %errors)

pi_list = []
for i in range(len(phase_z_list_l2l)):
    step = (2*np.pi)/len(phase_z_list_l2l[i])
    pl = np.linspace(0,2*np.pi,int(np.round((2*np.pi/step))), endpoint = False) # excludes 'stop' point 
    pi_list.append(pl) #pi value at each tidal point

flat_list_pi = [item for sublist in pi_list for item in sublist] #string sublists together
flat_list_phase_z = [item for sublist in phase_z_list_l2l for item in sublist] #string sublists together
flat_list_phase_t = [item for sublist in phase_t_list_l2l for item in sublist] #string sublists together
#%% Read in file containing iceqake detections from CNN

os.chdir('/Users/Kira/OneDrive/Data Science/My_DS_Projects/Icequake_CNN_post/')
file = 'DR14_CNN_icequake_detections.csv'
cnn_amps = pd.read_csv(file)

ev_list = cnn_amps.event
ev_list = [str(el) for el in ev_list]

#%% Read in file with pi-value (tidal phase) of each icequake

os.chdir('/Users/Kira/OneDrive/Data Science/My_DS_Projects/Icequake_CNN_post/')
infile = 'DR14_CNN_icequake_detections_pi_values.csv'

icequake_pi_values = []
with open(infile, 'r') as csvfile:
     csvreader = csv.reader(csvfile)
     for row in csvreader: 
        icequake_pi_values.append(row) 

# Omit icequakes that happen before tidal cycle "start" - otherwise many icequakes can be incorrectly associated with pi = 0. 
first_tide = flat_list_phase_t[0]
first_iq = datetime.strptime(icequake_pi_values[0][1], "%Y-%m-%d %H:%M:%S") 
for tt in range(len(icequake_pi_values)):
    r = icequake_pi_values[tt]
    iq_t = datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S") 
    if iq_t < first_tide:
        continue
    else:
        break

list_of_pi_lists = []
for s in range(tt, len(icequake_pi_values)): 
    r = icequake_pi_values[s]
    pi_value = r[-1]  
    pi_value = float(pi_value)
    list_of_pi_lists.append(pi_value)

#%% Read in STA/LTA detections
os.chdir('/Users/Kira/OneDrive/Data Science/My_DS_Projects/Icequake_CNN_post/')
infile = 'DR14_STALTA_icequake_detections_pi_values.csv'

sta_icequake_pi_values = []
with open(infile, 'r') as csvfile:
     csvreader = csv.reader(csvfile)
     for row in csvreader: 
        sta_icequake_pi_values.append(row) 


# Omit icequakes that happen before tidal cycle "start" - otherwise many icequakes can be incorrectly associated with pi = 0. 8/3/2020
first_tide = flat_list_phase_t[0]
first_iq = datetime.strptime(sta_icequake_pi_values[0][0], "%m/%d/%y %H:%M") 
for tt in range(len(sta_icequake_pi_values)):
    r = sta_icequake_pi_values[tt]
    iq_t = datetime.strptime(r[0], "%m/%d/%y %H:%M")
    iq_t = pd.to_datetime(iq_t)
    if iq_t < first_tide:
        continue
    else:
        break

# Omit icequakes that happen after certain date - Use this to match CNN detections if they are only X months long - 7/14/21
cutoff_date = datetime(2015,3,1,0,0,0)
for cc in range(len(sta_icequake_pi_values)):
    r = sta_icequake_pi_values[cc]
    iq_t = datetime.strptime(r[0], "%m/%d/%y %H:%M")
    # iq_t = datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S") #changed on 7/14/21
    if iq_t < cutoff_date:
        continue
    else:
        break
    
    
sta_list_of_pi_lists = []
for s in range(tt, cc + 1): 
    r = sta_icequake_pi_values[s]
    # pi_value = r[-1][0:-1]
    pi_value = r[-1]  # 7/31/2020 don't want to cut off last decimal place
    pi_value = float(pi_value)
    sta_list_of_pi_lists.append(pi_value)

#%% Plot figure of both CNN results and STA/LTA detections + tidal displacement

z=0

data1 = list_of_pi_lists #CNN
data2 = sta_list_of_pi_lists

plt.style.use('seaborn-white')
plt.rcParams['figure.figsize'] = 6,7
text_size = 12
tick_label_size = 10
fig, ax1 = plt.subplots(1,1)
nbins = 24 # Hourly bins
# nbins = 24*6  # 10-min bins 
bins = np.linspace(0,2*np.pi, nbins, endpoint = True) 

# Plot histograms of events at each tidal phase    
cnn_color = '#E15759' 
sta_color = 'grey'
a1 = ax1.hist(data1, bins = bins, color = cnn_color, align = 'mid', edgecolor = 'k', label = 'CNN')
ax1.hist(data2, bins = bins, color = sta_color, align = 'mid', edgecolor = 'k', label = 'Old Technique')
a2 = np.histogram(data1, bins = bins)
ax1.legend(loc = 'upper left')

# Plot tidal data    
ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
ax2.set_ylabel('               Vertical Tidal Displacement [m]', fontsize = text_size)
ax2.set_ylim(-2,1)
ax2.set_yticks([-1, 0, 1])
ax2.tick_params(axis= 'y',labelsize = tick_label_size)
ax2.tick_params(axis = 'both',which = 'major',length = 4, direction = 'out', pad = 0, width = 1, color = 'k', right = True, top = False)

for i in range(len(pi_list)):
    pi = pi_list[i]
    phase = phase_z_list_l2l[i]
    ax2.plot(pi, phase, c = 'steelblue', alpha = 0.03, linewidth = 3)   

ax1.set_xlabel('Tidal Phase', fontsize = text_size)
ax1.set_ylabel('Icequake Count', fontsize = text_size)
ax1.set_xlim(-0.1,6.38)

ax1.tick_params(axis= 'both',labelsize = tick_label_size)
ax1.yaxis.label.set_color('k')
ax2.yaxis.label.set_color('steelblue')
tick_locs = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
ax1.set_xticks(tick_locs)
labels = [0, r'$\pi/2$',r'$\pi$',r'$3\pi/2$', r'$2\pi$']
ax1.set_xticklabels(labels, horizontalalignment = 'right')         
ax1.tick_params(axis = 'both',which = 'major',length = 4, direction = 'out', pad = 0, width = 1, color = 'k', right = False, top = False)

#%% Build Convolutional Neural Network for detecting icequakes. 
# Training of CNN was done using 24,000 icequake + noise windows (each 8 s long), using computing resources from DeepThought2 cluster at University of Maryland, College Park
# Code below is included for reference only. Files to run this section of code are not included on GitHub due to size constraints. 
# Please contact me directly if you are interested in learning more about this project or running this code. - Kira

import keras
import os
import numpy as np
import obspy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from sklearn.model_selection import train_test_split
import pandas as pd



#######################################################
#######################################################
# Parameters

batch_size = 128
num_classes = 1 # 2 options: 1 = event or 0 = noise
num_epochs = 5 # Increase for more training

data_rows = 1 # 1 for a single time series, e.g., Z component
data_cols = 5 * 200 # seconds * sampling_rate = samples
input_shape = (data_rows, data_cols, 3) # last value is 3 for 3-component seismic data

# Shape for Convolutional layers needs to be (n_images, x_shape, y_shape, channels). Channels = 1 for greyscale, 3 for RGB.
#######################################################
#######################################################



#######################################################
#######################################################
# Read in data and set up arrays

# Data to train CNN on: 
n_events_used = 12000

# Input file: I
os.chdir('/Users/Kira/OneDrive/NASA/RIS_investigation/DATA/Filtered_Data/CNN_Data/DR14_windows/')
station = 'DR14'
event_I = 'DR14_events_to_keep_Full_List.txt' 
event_df = pd.read_csv(event_I, names = ['event'])
#######################################################
#######################################################



#######################################################
#######################################################
# Each event as an array like: np.array([[Z], [N], [E]])
s_list = []
errors = []
data_length = 8 * 200 # seconds * sampling_rate. This is the length of the data AS READ IN. Leave as is - even if later trimming to 5s windows. - 6/18/21
os.chdir('/Volumes/NASA_DATA/NASA_DATA/Filtered_RIS_Seismic_Data/CNN_Data/DR14_windows/8s_event_windows/')    
for i in range(n_events_used):
# for i in range(10):
    ev_name = str(event_df.event.iloc[i])
    
    try: 
        os.chdir(ev_name)   
   
        st = obspy.Stream()
        st += obspy.read('%s_*HHZ.sac' %ev_name)
        st += obspy.read('%s_*HHN.sac' %ev_name)
        st += obspy.read('%s_*HHE.sac' %ev_name)
        
        min_length = np.min([x.stats.npts for x in st])
        
        
        if min_length < data_length:
            print("%s too short, skipping. Min length: %s" % (ev_name, min_length))
            os.chdir('../.')
            continue
                    
        # Normalize each 3-comp event based on the absolute value max value observed on any component - Following Ross et al 2018b
        max_value = np.max([np.abs(x.data) for x in st])
        st[0].data = st[0].data/max_value
        st[1].data = st[1].data/max_value
        st[2].data = st[2].data/max_value
        
        ts1 = st[0].data[:-1] # trim to 1600 samples from 1601
        ts2 = st[1].data[:-1]
        ts3 = st[2].data[:-1]
        
        # Reshape to 2D arrays. 1 row, X columns (npts)
        ts1 = np.reshape(ts1, (1,ts1.shape[0]))
        ts2 = np.reshape(ts2, (1,ts2.shape[0]))
        ts3 = np.reshape(ts3, (1,ts3.shape[0]))
                
        s = np.dstack((ts1, ts2, ts3)) # 3D array
        
        s_list.append(s)
        os.chdir('../.')    
    
    except FileNotFoundError:
        errors.append(ev_name)
        print('%s no directory' %ev_name)

event_array = np.stack(s_list, axis = 0) # 4D. (n_images, x_shape, y_shape, channels)
print(event_array.shape)   
print(event_array.ndim)    
    
    
# Define noise windows
# Have equal numbers noise and event windows
os.chdir('/Users/Kira/OneDrive/NASA/RIS_investigation/DATA/Filtered_Data/CNN_Data/DR14_windows/')
noise_I = 'DR14_noise_windows_to_keep.txt' 
noise_df = pd.read_csv(noise_I, names = ['event'])

# Each event as an array like: np.array([[Z], [N], [E]])
x_train_noise = []
errors = []
os.chdir('/Volumes/NASA_DATA/NASA_DATA/Filtered_RIS_Seismic_Data/CNN_Data/DR14_windows/8s_noise_windows_NEW/')    
for i in range(n_events_used):
    ev_name = str(noise_df.event.iloc[i])
    try:
        os.chdir(ev_name)
 
        st = obspy.Stream()
        st += obspy.read('%s_*HHZ.sac' %ev_name)
        st += obspy.read('%s_*HHN.sac' %ev_name)
        st += obspy.read('%s_*HHE.sac' %ev_name)
           
        min_length = np.min([x.stats.npts for x in st])
                
        if min_length < data_length:
            print("%s too short, skipping. Min length: %s" % (ev_name, min_length))
            os.chdir('../.')
            continue
        
        # Normalize each 3-comp event based on the absolute value max value observed on any component
        max_value = np.max([np.abs(x.data) for x in st])
        st[0].data = st[0].data/max_value
        st[1].data = st[1].data/max_value
        st[2].data = st[2].data/max_value
        
        ts1 = st[0].data[:-1] # trim to 1600 samples from 1601
        ts2 = st[1].data[:-1]
        ts3 = st[2].data[:-1]
        
        # Reshape to 2D arrays
        ts1 = np.reshape(ts1, (1,ts1.shape[0]))
        ts2 = np.reshape(ts2, (1,ts2.shape[0]))
        ts3 = np.reshape(ts3, (1,ts3.shape[0]))
        
        n = np.dstack((ts1, ts2, ts3)) # 3D array
        
        x_train_noise.append(n)
        os.chdir('../.')    
  
    
    except FileNotFoundError:
        errors.append(ev_name)
        print('%s no directory' %ev_name)

noise_array = np.stack(x_train_noise, axis = 0) # 4D. (n_images, x_shape, y_shape, channels)
print(noise_array.shape)   
#######################################################
#######################################################    


#######################################################
#######################################################
# Combine event and noise windows into master x_train
combined_array = np.vstack((event_array, noise_array))

event_key = []
for i in range(len(event_array)):
    event_key.append(np.array([1])) # event = 1, noise = 0 
event_key = np.stack(event_key, axis = 0)   

noise_key = []
for i in range(len(noise_array)):
    noise_key.append(np.array([0]))
noise_key = np.stack(noise_key, axis = 0)   

combined_key = np.vstack((event_key, noise_key))
#######################################################
#######################################################



#######################################################
#######################################################
## Split out training vs validation datasets
x_train, x_test, y_train, y_test = train_test_split(combined_array, combined_key, test_size=0.25, shuffle = True)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#######################################################
#######################################################



#######################################################
#######################################################
## Set up CNN

model = Sequential() # Stepping thought layers sequentially
model.add(Conv2D(filters = 32, kernel_size=(1, 21), # Start with convolutional layer with 32 neurons (filters), each has 3x3 kernel size (height, width)
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1, 2))) #Add this layer next
model.add(Dropout(0.5)) #Dropout 50% of neurons

model.add(Conv2D(64, (1, 15), activation='relu')) #Next layer is a 64-kernel Convolutional layer
model.add(MaxPooling2D(pool_size=(1, 2))) #Add this layer next
model.add(Dropout(0.5)) #Dropout 50% of neurons

model.add(Conv2D(128, (1, 11), activation='relu')) #Next layer is a 128-kernel Convolutional layer
model.add(MaxPooling2D(pool_size=(1, 2))) #Add this layer next
model.add(Dropout(0.5)) #Dropout 50% of neurons

model.add(Conv2D(256, (1, 9), activation='relu')) #Next layer is a 256-kernel Convolutional layer
model.add(MaxPooling2D(pool_size=(1, 2))) #Add this layer next
model.add(Dropout(0.5)) #Dropout 50% of neurons

model.add(Flatten()) #Flattens from image to vector
model.add(Dense(200, activation='relu')) #Dense = fully connected neurons

model.add(BatchNormalization()) # Added 6/18/21
model.add(Dense(num_classes, activation='sigmoid')) #'softmax'

model.summary()
#######################################################
#######################################################



#######################################################
#######################################################
"""Now, train the network. You can play with optimizer methods and options..."""

opt = keras.optimizers.Adam() # Default learning_rate=0.001
model.compile(loss=keras.losses.binary_crossentropy, #cross entropy always used for class. probs. Cros entropy = log loss.
              optimizer= opt,
              metrics=['accuracy']) 

history = model.fit(x_train, y_train,
          batch_size=batch_size, # Number of samples per gradient update.
          epochs=num_epochs,
          verbose=1, #printing out as it trains
          validation_data=(x_test, y_test))

# evaluate the model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('[Accuracy] Train: %.3f, Test: %.3f' % (train_acc, test_acc))
#######################################################
#######################################################

# plot history
# plt.figure()
# plt.plot(history.history['accuracy'], label='train') # How well model is 'learning'
# plt.plot(history.history['val_accuracy'], label='test') # How well model is 'generalizing' (on new data)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


#######################################################
#######################################################
"""Save model"""
os.chdir('/Volumes/NASA_DATA/NASA_DATA/Filtered_RIS_Seismic_Data/CNN_Data/Saved_Models/')    

model.save("model_%sepochs_%stest_%strain.h5" %(num_epochs, x_test.shape[0], x_train.shape[0]))
#######################################################
#######################################################