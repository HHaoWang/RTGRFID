import numpy as np
import torch
from sklearn import metrics
from torch import nn

import Dataset
import utils
from RTGRFID import RTGRFID

dataDirectory = r"C:\Users\HHao\OneDrive\学习\研究生\小论文\实验数据\33"
numberOfRows = 3
numberOfCols = 3
tagArray2 = np.array([
    ['E280689400004013F1505D17', 'E280689400005013F1507117', 'E280689400005013F1506917'],
    ['E280689400004013F1507517', 'E280689400004013F155114C', 'E280689400005013F155154C'],
    ['E280689400004013F1506117', 'E280689400004013F1506D17', 'E280689400005013F1506517'],
])

tagArray1 = np.array([
    ['E280689400005013F1509918', 'E280689400005013F150B118', 'E280689400005013F150A518'],
    ['E280689400004013F1509D18', 'E280689400004013F150B518', 'E280689400005013F150A918'],
    ['E280689400005013F1509518', 'E280689400004013F150AD18', 'E280689400004013F150A118'],
])

seq_len = 100
epochs = 20

data, labels = Dataset.get_data(dataDirectory, tagArray1, tagArray2, seq_len)

print("数据集准备完毕\n")

model = RTGRFID(seq_len)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

idx = [i for i in range(len(data))]
np.random.shuffle(idx)
train_data_size = int(len(data) * 0.7)
train_data_idx = idx[:train_data_size]
valid_data_idx = idx[train_data_size:]

train_data = data[train_data_idx]
train_labels = labels[train_data_idx]
valid_data = data[valid_data_idx]
valid_labels = labels[valid_data_idx]

for epoch in range(epochs):
    idx = [i for i in range(len(train_data))]
    np.random.shuffle(idx)
    model.train()
    for i in range(len(idx)):
        x = train_data[idx[i]]
        y = torch.tensor(train_labels[idx[i]])
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Epoch: {}, Loss: {}'.format(epoch, loss))

    model.eval()
    with torch.no_grad():
        y_true = [utils.one_hot_to_string(label) for label in valid_labels]
        y_pred_proba = []
        y_pred_labels = []
        for i in range(len(valid_data)):
            x = valid_data[i]
            y = torch.tensor(valid_labels[i])
            y_pred = model(x)
            # y_pred_proba.append(y_pred)
            y_pred_labels.append(utils.one_hot_to_string(utils.convert_to_one_hot(y_pred)))

        accuracy = metrics.accuracy_score(y_true, y_pred_labels)
        print('Epoch:{},  Accuracy:{:.4f}'.format(epoch, accuracy))
