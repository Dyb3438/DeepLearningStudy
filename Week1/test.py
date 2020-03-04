import torch
import collections
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

x = torch.ones(1000, 2)
x0 = torch.normal(2 * x, 1)
y0 = torch.zeros(1000)
x1 = torch.normal(-2 * x, 1)
y1 = torch.ones(1000)
x = torch.cat((x0, x1), 0)
y = torch.cat((y0, y1)).type(torch.LongTensor)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.relu = torch.nn.ReLU()
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(2, 10, 2)
# seq1 = torch.nn.Sequential(collections.OrderedDict([
#     ('l1', torch.nn.Linear(2, 10)),
#     ('relu1', torch.nn.ReLU()),
#     ('l2', torch.nn.Linear(10, 2))
# ]))
# net = seq1

optimize = torch.optim.Adam(net.parameters(), lr=0.01)
loss = torch.nn.CrossEntropyLoss()

for i in range(1000):
    predict = net(x)
    los = loss(predict, y)
    optimize.zero_grad()
    los.backward()
    optimize.step()

softmax = torch.nn.Softmax(dim=1)
print(torch.max(softmax(net(x)), 1))
