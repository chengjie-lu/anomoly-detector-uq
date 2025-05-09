import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import random
import numpy as np

from deepluq import metrics


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10 * 1080, 5 * 1080)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(5 * 1080, 1080)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1080, 2)

    def forward(self, x):
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
        apply_dropout(self.dropout1)
        apply_dropout(self.dropout2)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


PATH = './anomaly_detector_50.pth'
model = Net()
model.load_state_dict(torch.load(PATH, weights_only=True, map_location=torch.device('cpu')))
model.eval()

# load normal dataset:
with open('./normal.pickle', 'rb') as fp:
    normal = pickle.load(fp)

normal = normal.reshape((30000, 10, 1080))
normal = normal / 3.5

uq_metrics = metrics.UQMetrics()

for i in range(1):
    rnd = random.randint(1, 30000)
    data = torch.tensor(normal[rnd], dtype=torch.float32)
    data = data.unsqueeze(0)

    logits = []
    for j in range(30):
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        outputs = nn.functional.softmax(outputs, dim=1)
        logits.append(outputs.data.tolist()[0])

    vr = uq_metrics.cal_vr(logits)
    se = uq_metrics.calcu_entropy(np.mean(np.transpose(logits), axis=1))
    mi = uq_metrics.calcu_mi(logits)

    uq_classification = {'variation ratio': vr, 'shannon entropy': se,
                         'mutual information': mi}
    print(uq_classification)
