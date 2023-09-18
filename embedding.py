import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
EMBEDDING_DIM=100
em_ep=2
batch_size = 64

tri=pd.read_csv('tri.csv')
voc=pd.read_csv('voc.csv')

vocab=voc['word'].tolist()
word_to_ix = {word: i for i, word in enumerate(vocab)}

w1=tri["word1"].tolist()
w2=tri["word2"].tolist()
w3=tri["word3"].tolist()
l=len(tri)
trigrams=[((w1[i],w2[i]),w3[i]) for i in range(l)]

# print(word_to_ix["of"],"\n",word_to_ix["movie"])

class emdataset(Dataset):
    def __init__(self, data):
        self.data=[]
        self.target=[]
        for context,target in data:
            context_idxs=[word_to_ix[w] for w in context]
            self.data.append(context_idxs)
            self.target.append(word_to_ix[target])
        self.len = len(data)
        self.data=torch.tensor(self.data).cuda()
        self.target=torch.tensor(self.target).cuda()
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    def __len__(self):
        return self.len
    
emtrain_data, emval_data = train_test_split(trigrams, test_size=0.2)
emtrain_loader = DataLoader(emdataset(emtrain_data), batch_size=batch_size, shuffle=True)
emval_loader = DataLoader(emdataset(emval_data), batch_size=batch_size, shuffle=True)

import torch.nn.functional as F
class Embeder(nn.Module):
    def __init__(self, vocab_size, embed_dim,context_size):
        super(Embeder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(context_size * embed_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embed(x).view(x.shape[0],-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        o= F.log_softmax(x,dim=1)
        return o
    
losses = []
loss_function = nn.NLLLoss().cuda()
model = Embeder(len(vocab), EMBEDDING_DIM, 2).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)
best_val_loss = 10086.0

#训练词嵌入
losses=[]
for epoch in range(em_ep):
    total_loss = 0
    # model.train()
    for context, target in emtrain_loader:
        optimizer.zero_grad()
        outputs= model(context)
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()

    # model.eval()
    val_loss=0.0
    with torch.no_grad():
        for context, target in emval_loader:   
            outputs= model(context)   
            val_loss += loss_function(outputs, target).item()
    val_loss = val_loss / len(emval_loader)
    print("Epoch:", epoch+1, "\tTrain Loss:", "%.5f"%loss.item(), "\tVal Loss:", "%.5f"%val_loss)
    losses.append([loss.item(), val_loss])
    if(val_loss < best_val_loss):
        best_val_loss = val_loss
        torch.save(model.state_dict(), "./model/embeder.pth")
losses=pd.DataFrame(losses,columns=["train_loss","val_loss"])
losses.to_csv("./model/embeder_loss.csv")


y_pred = []
y_test = []
model= Embeder(len(vocab), EMBEDDING_DIM, 2).cuda()
model.load_state_dict(torch.load("./model/embeder.pth"))
test_loader=DataLoader(emdataset(trigrams),batch_size=batch_size,shuffle=False)
with torch.no_grad():
    for inputs, target in test_loader:
        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)
        y_pred.extend(predicted.tolist())
        y_test.extend(target.tolist())

# 计算准确率和其他评估指标
accuracy = accuracy_score(y_test, y_pred)

print(f'准确率: {accuracy:.4f}')