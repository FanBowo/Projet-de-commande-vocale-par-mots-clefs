# -*- coding: utf-8 -*-
"""

"""
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import re
import os
from pandas import Series,DataFrame
from torch.utils.data import Dataset, DataLoader
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#%% Paramètres à modifier:
#%%	Chemin vers le dataset 
path_current='D://data_speech_commands_v0.02'
#Nombre de Labels qu'on veut reconnaitre, entre 1-35
NUM_CAT=10
#Pourcentage des enregistrements utilisés pour l'apprentissage
Train_pro=0.1
#Pourcentage des enregistrements utilisés pour la validation
Vali_pro=0.05
#Pourcentage des enregistrements utilisés pour le test
Test_pro=0.05


SR=16000    #Taux d'échantillonnage
N_FFT=1024 	#Taille de la fenêtre FFT
HOP_LENGTH=512 #Nombre d'échantillons entre deux trames successives
N_MELS=128 #Nombre de bandes de Mel générées

PROP_NOISE=0.05	#Proportion du bruit à ajouter aux enregistrements	 
	 
#Taille du Batch = Nombre d'enregistrements entrés en parallèle
BATCH_SIZE=1	 

#Paramètres du CNN	 
NUM_CONV1=12	#Profondeur de la première couche de convolution = Nombre des kernels de convolution de la première couche
HEIGHT_CONV1=5	#Longueur du filtre de la première couche
WIDTH_CONV1=5	#Largeur du filtre de la première couche

NUM_CONV2=30	#Profondeur de la deuxième couche de convolution = Nombre des kernels de convolution de la deuxième couche
HEIGHT_CONV2=5	#Longueur du filtre de la deuxième couche
WIDTH_CONV2=5	#Largeur du filtre de la deuxième couche

NUM_CONV3=60	#Profondeur de la triusième couche de convolution = Nombre des kernels de convolution de la troisième couche
HEIGHT_CONV3=2	#Longueur du filtre de la troisième couche
WIDTH_CONV3=2	#Largeur du filtre de la troisième couche

NUM_CONV4=120#numbre de coeurs de convolution pour la quatième couche

HIDDEN_SIZE= NUM_CONV4#On definit la taille de couche cachée même que celle de la dernière couche de convolution
INPUT_SIZE=NUM_CONV4#On defint la taille d'entrée de RNN même que celle de la dernière couche de convolution 
OUTPUT_SIZE=NUM_CAT#On defint la taillle de sortie mpême que lle nombres de types de sons

POOL_HEIGHT=2	#longueur de la couche de Pooling 
POOL_WIDTH=2	#Largeur de la couche de Pooling

EPOCH=10		#Nombre de répétitions d'apprentissage
LEARNING_RATE=0.001	#Learning rate 
MOMENTUM=0.9		#Momentum	 
	
#%%	
FilesListALL=[]	#Enregistrer tous les chemins de tous les fichiers
LabelsALL=[]	#Enregistrer tous les labels existants
FilesListNoise=[]	#Enregistrer les chemins vers les bruits fournis dans le dataset
LabelsNoise=[]		#Enregistrer les labels des bruits fournis dans le dataset



#%%
#Construction des listes des chemins et des labels
for dirpath, dirnames, filenames in os.walk(path_current):
    for file in filenames:
        if os.path.splitext(file)[1]=='.wav':
            temp=os.path.join(dirpath,file)
            #pour obtenir des labels des audios, ces sont le même que le nom de dossiers
            TempLabel=re.split(r'[\:,\//,\\]+',temp)[2]
            if TempLabel=='_background_noise_':
                FilesListNoise.append(temp)
                LabelsNoise.append(TempLabel)
            else:
                FilesListALL.append(temp)
                LabelsALL.append(TempLabel)
				
#%% 
# Liste des labels et nombre de labels
LabelsSet=set(LabelsALL)
LabelsSet=list(LabelsSet)
LabelsNum=len(LabelsSet)


#%% Construction d'une liste de taille 35, chaque entrée correspond à un label, 
###	chaque entrée est de taille l'nsemble des enregistrements du label, et contient
###	soit le chemin vers le fichier ou le label correspondant.
Audio_All_Path=[[],[],[],[],[],[],[],[],[],[],
                [],[],[],[],[],[],[],[],[],[],
                [],[],[],[],[],[],[],[],[],[],
                [],[],[],[],[]]
AudioLabel_All=[[],[],[],[],[],[],[],[],[],[],
                [],[],[],[],[],[],[],[],[],[],
                [],[],[],[],[],[],[],[],[],[],
                [],[],[],[],[]]

#Transfer des labels au nombres correspandants
LabelsDic = {}
for i in range(LabelsNum):
    LabelsDic[  LabelsSet[i]   ] = i

for i in range(len(LabelsALL)):
    labelID=LabelsDic[LabelsALL[i]]
    Audio_All_Path[labelID].append(FilesListALL[i])
#    print(FilesListALL[i])
    AudioLabel_All[labelID].append(LabelsDic[LabelsALL[i]])
#    print(LabelsALL[i])
#    print(LabelsDic[LabelsALL[i]])


#%%	Déclaration des listes utilisées pour l'apprentissage, la validation et le test
# Enregistrer les chemins et les labels des fiches utilisés pour l'apprentissage 
TrainFiles=[]
TrainLabels=[]

# Enregistrer les chemins et les labels des fiches utilisés pour la validation
ValiFiles=[]
ValiLabels=[]

# Enregistrer les chemins et les labels des fiches utilisés pour le test 
TestFiles=[]
TestLabels=[]

#%% Construction des listes des chemins et des labels des enregistrements utilisés en apprentissage, validation et test
for i in range(NUM_CAT):
    for j in range(math.floor(Train_pro*len(Audio_All_Path[i]))):
        TrainFiles.append(Audio_All_Path[i][j])
        TrainLabels.append(AudioLabel_All[i][j])
    
    #Calcul des indices du début et de fin de l'ensemble de validation
    begin=math.floor(Train_pro*len(Audio_All_Path[i]))
    end=math.floor(Train_pro*len(Audio_All_Path[i]))+\
                     math.floor(Vali_pro*len(Audio_All_Path[i]))
    #Construction de l'ensemble utilisé pour la validation 
    for j in range(begin,end):       
        ValiFiles.append(Audio_All_Path[i][j])
        ValiLabels.append(AudioLabel_All[i][j])
    
    #Calcul des indices du début et de fin de l'ensemble de test
    begin=math.floor(Train_pro*len(Audio_All_Path[i]))+\
                     math.floor(Vali_pro*len(Audio_All_Path[i]))
    end=math.floor(Train_pro*len(Audio_All_Path[i]))+\
                     math.floor(Vali_pro*len(Audio_All_Path[i]))+\
                     math.floor(Test_pro*len(Audio_All_Path[i]))
    
	#Construction de l'ensemble utilisé pour la validation 
    for j in range(begin,end):
       TestFiles.append(Audio_All_Path[i][j])
       TestLabels.append(AudioLabel_All[i][j])
    

    
#%%
#Dimensions du spectrogramme
WIDTH_SPEC=math.ceil(SR/HOP_LENGTH)
HEIGHT_SPEC=N_MELS


#%% Fonction chargement des enregistrements audio
def default_loader(path):
    # Lecture des enregistrements audio
    y,sr =  librosa.load(path,sr=SR)  
    #len(y)=T_audio_seconds*sr
    #len(logmelspec)=n_mels
    #len(logmelspec[0,:])=math.ceil(len(y)/hop_length)
    
    #Ajout du bruit aux audios avec une longueur < 1s
    if len(y)<sr:	#longueur des audios < 1s
        #Calcul de la longueur qu'il faut ajouter aux audios
        Half_Len_Audio=math.ceil((sr-len(y))/2)
        temp_audio=np.zeros(sr)
        
        #Lecture aléatoire d'un enregistrement d'un bruit
        y1,sr1=librosa.load(FilesListNoise[np.random.randint(0,len(FilesListNoise))],sr=SR)  
        
        #Premier morceau du bruit aléatoire à ajouter 
        RandStart=np.random.randint(0,len(y1)-Half_Len_Audio)
		#Ajout du premier morceau du bruit
        temp_audio[0:Half_Len_Audio]=y1[RandStart:RandStart+Half_Len_Audio]
        #Construction de 0.5s
        temp_audio[Half_Len_Audio:Half_Len_Audio+len(y)]=y
        #Deuxième morceau du bruit aléatoire à ajouter
        RandStart=np.random.randint(0,len(y1)-Half_Len_Audio)
		#Ajout du deuxième morceau du bruit
        temp_audio[(sr-Half_Len_Audio):sr]=y1[RandStart:RandStart+Half_Len_Audio]
        #Construction de l'audio avec longueur = 1s complété par le bruit
        y=temp_audio
		
    #Cas où la longueur est > 1s ... n'est pas utilisée dans le dataset fourni    
    elif len(y)>sr:
        temp_audio=np.zeros(sr)
        Half_Len_Audio=math.ceil((len(y)-sr)/2)
        temp_audio=y[Half_Len_Audio:Half_Len_Audio+sr]
        y=temp_audio
    
    #Lecture aléatoire d'un bruit
    y2,sr2=librosa.load(FilesListNoise[np.random.randint(0,len(FilesListNoise))],sr=SR)     
    RandStart=np.random.randint(0,len(y2)-len(y))
    #Ajout de la proportion fixée du bruit, pour constituer l'audio final bruité
    y=y+PROP_NOISE*y2[RandStart:RandStart+len(y)]
    
    #Construction des spectrogrammes
    melspec = librosa.feature.melspectrogram(y, SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    logmelspec = librosa.power_to_db(melspec)
    Audio_tensor = torch.FloatTensor(logmelspec)
    Audio_tensor=Audio_tensor.view(1,HEIGHT_SPEC,WIDTH_SPEC)
    
    return Audio_tensor

#%% Chargement des données d'apprentissage, de validation et de test    
# Définition des dataloaders qui vont être appelés par la suite
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        
        self.audios = TrainFiles
        self.target = TrainLabels
        self.loader = loader

    def __getitem__(self, index):
        fn = self.audios[index]
        aud= self.loader(fn)
        target = self.target[index]
        #print(target)
        return aud,target

    def __len__(self):
        return len(self.audios)

class valiset(Dataset):
    def __init__(self, loader=default_loader):
        self.audios = ValiFiles
        self.target = ValiLabels
        self.loader = loader

    def __getitem__(self, index):
        fn = self.audios[index]
        aud= self.loader(fn)
        target = self.target[index]
        return aud,target

    def __len__(self):
        return len(self.audios)
    
class testset(Dataset):
    def __init__(self, loader=default_loader):
        #init  the path of data
        self.audios = TestFiles
        self.target = TestLabels
        self.loader = loader

    def __getitem__(self, index):
        fn = self.audios[index]
        aud= self.loader(fn)
        target = self.target[index]
        return aud,target

    def __len__(self):
        return len(self.audios)


#%%Appel des dataloaders
train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True)

vali_data  = valiset()
valiloader = DataLoader(vali_data, batch_size=BATCH_SIZE,shuffle=True)

test_data  = testset()
testloader = DataLoader(test_data, batch_size=BATCH_SIZE,shuffle=True)


#%% Tailles après les différentes couches
#Longueur et largeur après chaque couche
SIZE_AFTER_CONV1=[math.floor(HEIGHT_SPEC+1-HEIGHT_CONV1),\
                  math.floor(WIDTH_SPEC+1-WIDTH_CONV1)]
SIZE_AFTER_CONV1_POOL=[math.floor(SIZE_AFTER_CONV1[0]/POOL_HEIGHT),\
                       math.floor(SIZE_AFTER_CONV1[1]/POOL_WIDTH)]
SIZE_AFTER_CONV2=[math.floor(SIZE_AFTER_CONV1_POOL[0]+1-HEIGHT_CONV2),\
                  math.floor(SIZE_AFTER_CONV1_POOL[1]+1-WIDTH_CONV2)]
SIZE_AFTER_CONV2_POOL=[math.floor(SIZE_AFTER_CONV2[0]/POOL_HEIGHT),\
                       math.floor(SIZE_AFTER_CONV2[1]/POOL_WIDTH)]
SIZE_AFTER_CONV3=[math.floor(SIZE_AFTER_CONV2_POOL[0]+1-HEIGHT_CONV3),\
                  math.floor(SIZE_AFTER_CONV2_POOL[1]+1-WIDTH_CONV3)]


#%% Nombre d'éléments pour la première couche linéaire
NUM_ELEMENT_FIRST_LAYER_LINEAR=NUM_CONV3 * SIZE_AFTER_CONV3[0] *SIZE_AFTER_CONV3[1]


#%%Si une GPU est détectée
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Définition des fonctions utilisées par la suite: convolution + Activation + Fonction linéaire
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, NUM_CONV1, [HEIGHT_CONV1,WIDTH_CONV1])
        self.pool = nn.MaxPool2d(POOL_HEIGHT, POOL_WIDTH)
        self.conv2 = nn.Conv2d(NUM_CONV1, NUM_CONV2, [HEIGHT_CONV2,WIDTH_CONV2])
        self.conv3 = nn.Conv2d(NUM_CONV2, NUM_CONV3, [HEIGHT_CONV3,WIDTH_CONV3])
        self.conv4 = nn.Conv2d(NUM_CONV3, NUM_CONV4, [SIZE_AFTER_CONV3[0],1])
        
    def forward(self, x):
        #révolution -> activation -> pooling
        x = self.pool(F.relu(self.conv1(x)))
        #révolution -> activation -> pooling
        x = self.pool(F.relu(self.conv2(x)))
        #révolution -> activation -> pooling
        x = F.relu(self.conv3(x))
        #
        x = F.relu(self.conv4(x))
        return x

#%%	Déclaration du réseau
net = Net()
#%%	Si GPU détecté
net.to(device)

#%% Définition la structure de RNN
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.hidden_size = HIDDEN_SIZE

        self.i2h = nn.Linear(INPUT_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)
        self.i2o = nn.Linear(INPUT_SIZE + HIDDEN_SIZE, OUTPUT_SIZE)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #Combine la couche d'entrée et la couche cachée
        combined = torch.cat((input, hidden), 1)
        
        #Calculer la couche cachée
        hidden = self.i2h(combined)
        
        #Calciler la couche de sortie 
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
#%%	Déclaration du réseau
rnn_net=RNN()
#%%	Si GPU détecté
rnn_net.to(device)

#%%	Méthode d'optimisation utilisée
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

#%%	Apprentissage 
runningloss=[]
for epoch in range(EPOCH):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels= data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        CNNoutputs= net(inputs)
        
        #Initialiser la couche cachée
        hidden = rnn_net.initHidden()
        
        #Calculer la sortie, simutanément, mettre à jour la couche cachée
        for Index in range(CNNoutputs.size()[3]):
            outputs, hidden = rnn_net(CNNoutputs[:,:,0,Index], hidden)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 20 mini-batches
#            print('[%d, %5d] loss: %.3f' %
#                  (epoch + 1, i + 1, running_loss / 20))
            runningloss.append(running_loss / 100)
            running_loss = 0.0

print('Finished Training')
plt.figure()
plt.plot(runningloss)
plt.title('CNN+RNN')
plt.show()           

#torch.save(net,'D:\\google_drive_ori\\paris sud\\D2 Reconnaissance de pa parole\\Model1701.pth')



#%%Partie de validation
#Calcul du taux de reconnaissance global
correct = 0
total = 0
with torch.no_grad():
    for data in valiloader:
        #importation des enregistrements de l'ensemble construit de validation
        inputs, labels= data
        inputs, labels = inputs.to(device), labels.to(device)
        
        CNNoutputs= net(inputs)
        # forward + backward + optimize
        #Initialiser la couche cachée
        hidden = rnn_net.initHidden()
        #Calculer la sortie, simutanément, mettre à jour la couche cachée
        for Index in range(CNNoutputs.size()[3]):
            outputs, hidden = rnn_net(CNNoutputs[:,:,0,Index], hidden)
            
        #prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #Calcul de la somme des prédictions correctes
        correct += (predicted == labels).sum().item()

#%% Affichage du taux de reconnaissance global à partir de la phase de validation
print('Accuracy of the network on the %d test audios: %d %%' % (total,\
    100 * correct / total))

#%% Calcul du taux de reconnaissance pour chaque label après la phase de validation
class_correct = list(0. for i in range(NUM_CAT))
class_total = list(0. for i in range(NUM_CAT))
with torch.no_grad():
    for data in valiloader:
        inputs, labels= data
        inputs, labels = inputs.to(device), labels.to(device)
        
        CNNoutputs= net(inputs)
        # forward + backward + optimize
        #Initialiser la couche cachée
        hidden = rnn_net.initHidden()
        #Calculer la sortie, simutanément, mettre à jour la couche cachée
        for Index in range(CNNoutputs.size()[3]):
            outputs, hidden = rnn_net(CNNoutputs[:,:,0,Index], hidden)
        
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        if len(labels)>1:
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        else:
            label = labels
            class_correct[label] += c
            class_total[label] += 1


#%% Affichage du taux de reconnaissance pour chaque label après la phase de validation
for i in range(NUM_CAT):
    print('Accuracy of %5s : %2d %%' % (
        LabelsSet[i], 100 * class_correct[i] / class_total[i]))
		
  
#%% Partie du test
# Calcul du taux de reconnaissance global pour la phase du test       
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels= data
        inputs, labels = inputs.to(device), labels.to(device)
        
        CNNoutputs= net(inputs)
        # forward + backward + optimize
        #Initialiser la couche cachée
        hidden = rnn_net.initHidden()
        #Calculer la sortie, simutanément, mettre à jour la couche cachée
        for Index in range(CNNoutputs.size()[3]):
            outputs, hidden = rnn_net(CNNoutputs[:,:,0,Index], hidden)
            
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

#%% Affichage du taux de reconnaissance global pour la phase du test 
print('Accuracy of the network on the %d test audios: %d %%' % (total,\
    100 * correct / total))


#%% Calcul du taux de reconnaissance pour chaque label après la phase de test
class_correct = list(0. for i in range(NUM_CAT))
class_total = list(0. for i in range(NUM_CAT))
with torch.no_grad():
    for data in testloader:
        inputs, labels= data
        inputs, labels = inputs.to(device), labels.to(device)
        
        CNNoutputs= net(inputs)
        # forward + backward + optimize
        #Initialiser la couche cachée
        hidden = rnn_net.initHidden()
        #Calculer la sortie, simutanément, mettre à jour la couche cachée
        for Index in range(CNNoutputs.size()[3]):
            outputs, hidden = rnn_net(CNNoutputs[:,:,0,Index], hidden)
            
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        if len(labels)>1:
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        else:
            label = labels
            class_correct[label] += c
            class_total[label] += 1

#%% Affichage du taux de reconnaissance pour chaque label après la phase du test 
for i in range(NUM_CAT):
    print('Accuracy of %5s : %2d %%' % (
        LabelsSet[i], 100 * class_correct[i] / class_total[i]))