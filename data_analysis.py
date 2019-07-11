# -*- coding: utf-8 -*-
"""

@author: CHGHAF
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import re
import os
from os.path import isdir, join
from pathlib import Path
from pandas import Series,DataFrame
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import IPython.display as ipd

import math
import torch
#%%
## Dataset analysis
#%% path declaration
path='C:/Users/moham/Downloads/data_speech_commands_v0.02/'
#%% number of labels
labels = [f for f in os.listdir(path) if isdir(join(path, f))]
labels.sort()
print('Number of labels is: ' + str(len(labels)))
##%% courbe nombre de labels
#audios = []
#for lbl in labels:
#    audio = [f for f in os.listdir(join(path, lbl)) if f.endswith('.wav')]
#    audios.append(len(audio))
## Plot
#plt.figure(figsize=(12, 4))
#plt.plot(labels, audios)
#plt.xticks(labels, rotation='vertical')
#plt.yscale('linear')
#plt.title('Nombre des enregistrements par label')
#plt.grid(True)

#%% definition du fast fourier transform
def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    return xf, vals


#%% différence entre lecteurs
filenames = ['backward/0a2b400e_nohash_3.wav', 'backward/1a0f9c63_nohash_0.wav']
for filename in filenames:
    sample_rate, samples = wavfile.read(str(path) + filename)
    xf, vals = custom_fft(samples, sample_rate)
    plt.figure(figsize=(12, 4))
    plt.title('FFT du lecteur ' + filename[9:17] + ' pour le mot ' + filename[0:8])
    plt.plot(xf, vals)
    plt.xlabel('Fréquences')
    plt.grid()
    plt.show()
    
    
    
##%% waveplots
#y, sr = librosa.core.load(path + 'backward/0a2b400e_nohash_3.wav', sr=16000)
#D=np.abs(librosa.stft(y))
#plt.figure(figsize=(12, 4))   
#librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')
#plt.title('Power spectrogram')
#plt.colorbar(format='%+2.0f dB')
#plt.tight_layout() 
#
#plt.figure(figsize=(12, 4))   
#librosa.display.waveplot(y, sr)

#plt.figure(figsize=(12, 4))
#plt.plot(xf, vals)
#plt.xlabel('Fréquences')
#plt.grid()
#plt.show()  
    
##%% Nombre des enregistrements selon longueur
#nb_moins = 0
#nb_plus = 0
#nb_1 = 0
#moins=[]
#plus=[]
#egal=[]
#print('debut comptage')
#for label in labels:
#    audios = [f for f in os.listdir(join(path, label)) if f.endswith('.wav')]
#    for audio in audios:
#        sample_rate, samples = wavfile.read(path + label + '/' + audio)
#        if samples.shape[0] < sample_rate:
#            nb_moins += 1
#            moins.append(label+'/'+audio)
#        if samples.shape[0] > sample_rate:
#            nb_plus += 1
#            plus.append(label+'/'+audio)
#        elif samples.shape[0] == sample_rate:
#            nb_1 += 1
#            egal.append(label+'/'+audio)
#print('Nombre des enregistrements inférieurs à 1s ' + str(nb_moins))
#print('Nombre des enregistrements supérieurs à 1s ' + str(nb_plus))
#print('Nombre des enregistrements égaux à 1s ' + str(nb_1))
#
#print('fin comptage')

#%% fonction log des valeurs du spectrogramme
def log_specgram(audio, sample_rate, window_size=25,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

#%% figures enregistrement + spectrogramme + Mel spectrogramme + MFCC
filenames = ['backward/0ba018fc_nohash_0.wav', 'five/0a2b400e_nohash_2.wav']
for filename in filenames:
    sample_rate, samples = wavfile.read(str(path) + filename)
    y, SR = librosa.load(path + filename)

    freqs, times, spectrogram = log_specgram(samples, sample_rate)
    S = librosa.feature.melspectrogram(y, sr=SR, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=12)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)


    fig = plt.figure(figsize=(14, 14))
    
    ax1 = fig.add_subplot(411)
    ax1.set_title('Enregistrement : ' + filename)
    ax1.set_ylabel('Amplitude')
    ax1.set_ylabel('Temps en secondes')
    ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

    ax2 = fig.add_subplot(412)
    ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
           extent=[0, 1, freqs.min(), freqs.max()])
    ax2.set_yticks(freqs[::14])
    ax2.set_xticks(times[::4])
    ax2.set_title('Spectrogramme de l''enregistrement :' + filename)
    ax2.set_ylabel('Fréquences en Hz')
    ax2.set_xlabel('Temps en secondes')
    
    ax3 = fig.add_subplot(413)
    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
    ax3.set_title('Mel-spectrogramme de l''enregistrement : ' + filename)
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    
    ax4 = fig.add_subplot(414)
    librosa.display.specshow(delta2_mfcc, x_axis='time')
    ax4.set_ylabel('Coefficients MFCC')
    ax4.set_xlabel('Temps en secondes')
    ax4.set_title('MFCC')
    plt.colorbar()
    plt.tight_layout()
#%% audio backwards non complets
filenames = ['backward/0a396ff2_nohash_0.wav', 'backward/0c540988_nohash_0.wav', 'backward/0a2b400e_nohash_0.wav']
y1, sr1 = librosa.load(path + filenames[0], duration=1)
y2, sr2 = librosa.load(path + filenames[1], duration=1)
y3, sr3 = librosa.load(path + filenames[2], duration=1)


fig = plt.figure(figsize=(14, 16))

plt.subplot(3, 1, 1)
librosa.display.waveplot(y1, sr1)
plt.title('Audio :' + filenames[0])

plt.subplot(3, 1, 2)
librosa.display.waveplot(y2, sr2)
plt.title('Audio :' + filenames[1])

plt.subplot(3, 1, 3)
librosa.display.waveplot(y3, sr3)
plt.title('Audio :' + filenames[2])

s1=librosa.util.fix_length(s1, size=16000)

ax1 = fig.add_subplot(311)
ax1.set_title('Enregistrement : ' + filenames[0])
ax1.set_ylabel('Amplitude')
ax1.set_xlabel('Temps en secondes')
ax1.plot(np.linspace(0, s_r1/len(s1), s_r1), s1)
    


ax2 = fig.add_subplot(312)
ax2.set_title('Enregistrement : ' + filenames[1])
ax2.set_ylabel('Amplitude')
ax2.set_xlabel('Temps en secondes')
ax2.plot(np.linspace(0, s_r2/len(s2), s_r2), s2)    
    
ax3 = fig.add_subplot(313)
ax3.set_title('Enregistrement : ' + filenames[2])
ax3.set_ylabel('Amplitude')
ax3.set_xlabel('Temps en secondes')
ax3.plot(np.linspace(0, s_r3/len(s3), s_r3), s3)   

#%% Liste des labels

FilesListALL=[]# Pour enregestre des Paths de touts les fiches
LabelsALL=[]# Pour enregestre des Labels de touts les fiches
FilesListNoise=[]#Pour enregestre des Paths de touts les bruits
LabelsNoise=[]# Pour enregestre des Labels de touts les bruits
path_current='C:\\Users\\moham\\Downloads\\data_speech_commands_v0.02'

#Cet partie est pour constuire les quatres lists on a dit avant
for dirpath, dirnames, filenames in os.walk(path_current):
    for file in filenames:
        if os.path.splitext(file)[1]=='.wav':
            temp=os.path.join(dirpath,file)
            TempLabel=re.split(r'[\:,\//,\\]+',temp)[5]#pour obtenir des labels des audios, ces sont le même que le nom de dossiers
            if TempLabel=='_background_noise_':
                FilesListNoise.append(temp)
                LabelsNoise.append(TempLabel)
            else:
                FilesListALL.append(temp)
                LabelsALL.append(TempLabel)

LabelsSet=set(LabelsALL)
LabelsSet=list(LabelsSet)
LabelsNum=len(LabelsSet)


#%% Ajout du bruit
SR=16000
PROP_NOISE=0.05

#files = ['backward/42398aab_nohash_0.wav', 'backward/82951cf0_nohash_0.wav', 'backward/1c76f5f3_nohash_0.wav', 'backward/88d009d2_nohash_1.wav', 'backward/0cb74144_nohash_2.wav' ]
files = ['backward/42398aab_nohash_0.wav', 'backward/82951cf0_nohash_0.wav', 'backward/0cb74144_nohash_2.wav' ]

#fig = plt.figure(figsize=(14, 25))
fig = plt.figure(figsize=(14, 18))

for file in files:
#    os.system(path + file)
    y,sr =  librosa.load(path + file,sr=SR)
    plt.subplot(3, 1, files.index(file)+1)
    librosa.display.waveplot(y, sr=SR)
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.title('Audio original de :' + file)
    


#fig = plt.figure(figsize=(14, 25))
fig = plt.figure(figsize=(14, 18))

for file in files:
    y,sr =  librosa.load(path + file,sr=SR)
    
    if len(y)<sr:#length of audio shorter than 1 s
        
        Half_Len_Audio=math.ceil((sr-len(y))/2)
        temp_audio=np.zeros(sr)
        
        y1,sr1=librosa.load(FilesListNoise[np.random.randint(0,len(FilesListNoise))],sr=SR)  
        
        RandStart=np.random.randint(0,len(y1)-Half_Len_Audio)
        temp_audio[0:Half_Len_Audio]=y1[RandStart:RandStart+Half_Len_Audio]
        
        temp_audio[Half_Len_Audio:Half_Len_Audio+len(y)]=y
        
        RandStart=np.random.randint(0,len(y1)-Half_Len_Audio)
        temp_audio[(sr-Half_Len_Audio):sr]=y1[RandStart:RandStart+Half_Len_Audio]
        
        y=temp_audio
        
    elif len(y)>sr:#length of audio longer than 1 s
        temp_audio=np.zeros(sr)
        Half_Len_Audio=math.ceil((len(y)-sr)/2)
        temp_audio=y[Half_Len_Audio:Half_Len_Audio+sr]
        y=temp_audio  
        
    #lire des bruit
    y2,sr2=librosa.load(FilesListNoise[np.random.randint(0,len(FilesListNoise))],sr=SR)     
    RandStart=np.random.randint(0,len(y2)-len(y))
    #Ajouter des bruits
    y=y+PROP_NOISE*y2[RandStart:RandStart+len(y)]
    
    librosa.output.write_wav('C:/Users/moham/Downloads/data_speech_commands_v0.02/'+ str(files.index(file)) +'.wav', y, SR)
#    os.system(path+str(files.index(file))+'.wav')
    plt.subplot(3, 1, files.index(file)+1)
    librosa.display.waveplot(y, sr=SR)
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    plt.title('Audio bruité de :' + file +' -- Le taux du bruit est : %.02f' %PROP_NOISE)