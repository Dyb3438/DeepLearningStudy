import torch.nn as nn
import torch
import torch.utils.data as Data
import torchvision
from torch.optim import lr_scheduler, Adam
from os.path import exists
from os import mkdir

# define parameters
OFFSET_EPOCH = 0
EPOCH = 10
BATCH_SIZE = 100
VALIDATION_SIZE = 1000
INIT_LR = 0.0005
DOWNLOAD_MNIST = False
MNIST_ROOT = './../MNIST'
PKL_FILE = './pkl/cnn_net_params.pkl'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=7,
                padding=3
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.l1 = nn.Sequential(
            nn.Linear(
                in_features=64 * 3 * 3,
                out_features=50
            ),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.l2 = nn.Sequential(
            nn.Linear(
                in_features=50,
                out_features=10
            ),
            nn.Sigmoid()
        )
        return

    def forward(self, input):
        c1 = self.c1(input)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        l1 = self.l1(torch.flatten(c5, start_dim=1))
        l2 = self.l2(l1)
        return l2


def train(train_data, validation_data, device):
    validation_data = [i[:VALIDATION_SIZE] for i in validation_data]
    cnn = CNN().to(device)
    if exists(PKL_FILE):
        cnn.load_state_dict(torch.load(PKL_FILE))
        print('加载旧的参数成功')
    optimizer = Adam(cnn.parameters(), lr=INIT_LR)
    loss_fun = nn.CrossEntropyLoss()

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (0.9 ** (epoch + OFFSET_EPOCH)) if (0.9 ** (
            epoch + OFFSET_EPOCH)) > 1e-5 else 1e-5)

    for epoch in range(EPOCH):
        scheduler.step(epoch)
        for step, (image, label) in enumerate(train_data):
            # put data into device (gpu)
            image, label = image.to(device), label.to(device)
            predict = cnn(image)
            loss = loss_fun(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                cnn.eval()
                test_out = cnn(validation_data[0])
                test_predict_index = torch.max(test_out, 1)[1].detach().squeeze()
                train_predict_index = torch.max(predict, 1)[1].detach().squeeze()
                acc = (test_predict_index == validation_data[1]).sum().item() / validation_data[1].size(0)
                train_acc = sum(train_predict_index == label).item() / label.size(0)
                print('EPOCH:', epoch, '| STEP:', step, '| TRAIN_ACC: %.4f' % train_acc, '| VALI_ACC: %.4f' % acc,
                      '| LOSS: %.4f' % loss.item())
                cnn.train()
        torch.save(cnn.state_dict(), PKL_FILE)
        print('保存参数成功', '当前学习率:', optimizer.state_dict()['param_groups'][0]['lr'])


def test(test_data, device):
    cnn = CNN().to(device)
    cnn.eval()
    if exists(PKL_FILE):
        cnn.load_state_dict(torch.load(PKL_FILE))
        print('加载旧的参数成功')
    data = Data.DataLoader(test_data[0], batch_size=BATCH_SIZE)
    label = Data.DataLoader(test_data[1], batch_size=BATCH_SIZE)

    total = 0
    acc = 0
    for batch_data, batch_label in zip(data, label):
        test_out = cnn(batch_data)
        test_predict_index = torch.max(test_out, 1)[1].data.squeeze()
        acc += (test_predict_index == batch_label).sum().item()
        total += batch_data.size(0)
    total_acc = acc / total
    print('ACC: %.4f' % total_acc)
    return



def main(device):
    # load mnist
    train_data_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(28, padding=4),
        torchvision.transforms.RandomRotation(25),
        torchvision.transforms.ToTensor()
    ])
    train_data = torchvision.datasets.MNIST(MNIST_ROOT, train=True, download=DOWNLOAD_MNIST,
                                            transform=train_data_transform)
    test_data = torchvision.datasets.MNIST(MNIST_ROOT, train=False)

    # put mnist into Loader
    train_data = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data_data = (torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor) / 255.).to(device)
    test_data_labels = test_data.test_labels.to(device)
    test_data = [test_data_data, test_data_labels]

    # train
    # train(train_data, test_data, device)

    # test
    test(test_data, device)



if __name__ == '__main__':
    if exists('./pkl') is False:
        mkdir('./pkl')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)
