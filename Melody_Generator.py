import numpy as np 
from collections import Counter
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(7)
from music21 import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns
import os

filepath = "train_data/"
midis= []
for i in os.listdir(filepath):
    if i.endswith(".mid"):
        try:
            tr = filepath+i
            print(tr)
            midi = converter.parse(tr)
            midis.append(midi)
        except:
            print("err: ", filepath+i)

def get_melody(file):
    notes = []
    pick = None
    for j in file:
        songs = instrument.partitionByInstrument(j)
        try:
            part = songs.parts[1]
        except:
            part = songs.parts[0]
        pick = part.recurse()
        for element in pick:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
    return notes

corpus = get_melody(midis)

corpus_cnt = Counter(corpus)
Notes = list(corpus_cnt.keys())
rare_note = []
for index, (key, value) in enumerate(corpus_cnt.items()):
    if value < (max(list(corpus_cnt.values())) / 100):
        rare_note.append(key)
new_corpus = [element for element in corpus if element not in rare_note]

symb = sorted(list(set(new_corpus)))

mapping = {c: i for i, c in enumerate(symb)}
reverse_mapping = {i: c for i, c in enumerate(symb)}

length = 40
features = []
targets = []
numerical_corpus = np.array([mapping[j] for j in new_corpus])

features = np.lib.stride_tricks.sliding_window_view(numerical_corpus, length)
features = np.delete(features, -1, axis=0)
targets = numerical_corpus[length:]
data_len = len(targets)

X = (np.reshape(features, (data_len, length, 1))) / float(len(symb))
num_classes = len(np.unique(targets))
y = np.eye(num_classes)[targets]

X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)

tensor_X_train = torch.Tensor(np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))) 
tensor_y_train = torch.Tensor(y_train)
tensor_X_seed = torch.Tensor(np.reshape(X_seed, (X_seed.shape[0], X_seed.shape[1]))) 
tensor_y_seed = torch.Tensor(y_seed)

my_dataset = TensorDataset(tensor_X_train,tensor_y_train)
my_dataloader = DataLoader(my_dataset) 

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
input_size = X.shape[1]
hidden_size = 512
output_size = y.shape[1]
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = LSTMModel(input_size, hidden_size, output_size)
model.to(device)
update_time = 10

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)

total_losses = []
for epoch in range(0, 100):
    total_loss = 0
    tmp_loss = 0
    cnt = 0
    for data_x, data_y in my_dataloader:
        optimizer.zero_grad()
        d_x = data_x.to(device)
        d_y = data_y.to(device)
        output = model(d_x)
        loss = criterion(output, d_y)
        total_loss += loss
        tmp_loss += loss
        cnt += 1
        del(d_x)
        del(d_y)
        if cnt > update_time:
            avg_loss = tmp_loss / update_time
            avg_loss.backward()
            optimizer.step()
            tmp_loss = 0
            cnt = 0
            
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, total_loss / X_train.shape[0]))
    total_losses.append(total_loss.item())

def music_maker(Note_Count):
    model.eval()
    seed = tensor_X_seed[np.random.randint(0, X_seed.shape[0] - 1)]
    Music = ""
    Notes_Generated = []

    with torch.no_grad():
        for _ in range(Note_Count):
            seed = seed.to(device)
            seed = seed.unsqueeze(0)
            prediction = model(seed)          
            index = torch.argmax(prediction).item()
            index_N = index / float(len(symb))
            Notes_Generated.append(index)
            Music = [reverse_mapping[char] for char in Notes_Generated]
            seed = torch.cat((seed.squeeze(0), torch.tensor([index_N]).to(device)))
            seed = seed[1:]
    return Music

Music_notes = music_maker(100)
print(Music_notes)

def show(music):
    display(Image(str(music.write("lily.png"))))
    
def chords_n_notes(Snippet):
    Melody = []
    offset = 0 #Incremental
    for i in Snippet:
        note_snip = note.Note(i)
        note_snip.offset = offset
        Melody.append(note_snip)
        offset += 1
    Melody_midi = stream.Stream(Melody)   
    return Melody_midi

Melody_Snippet = Music_notes
Melody = chords_n_notes(Melody_Snippet)

show(Melody)
Melody_midi = stream.Stream(Melody) 
Melody.write('midi','Melody_Generated.mid')