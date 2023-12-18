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

seq_len = 90
epochs = 50

(train_data, train_labels), (valid_data, valid_labels), classes = Dataset.get_data(dataDirectory, tagArray1, tagArray2, seq_len,
                                                                                   0.3)

print("数据集准备完毕\n")

model = RTGRFID(seq_len, len(classes))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

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
        y_true_labels = [utils.one_hot_to_string(label) for label in valid_labels]
        y_pred_proba = []
        y_pred_labels = []
        for i in range(len(valid_data)):
            x = valid_data[i]
            y = torch.tensor(valid_labels[i])
            y_pred = model(x)
            y_pred_proba.append(y_pred)
            y_pred_labels.append(utils.one_hot_to_string(utils.convert_to_one_hot(y_pred)))

        accuracy = metrics.accuracy_score(y_true_labels, y_pred_labels)
        print(f'Epoch:{epoch + 1:00},  Accuracy:{accuracy:.4f}')
