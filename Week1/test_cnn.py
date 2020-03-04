import torch.nn as nn
import torch
import torch.utils.data as Data
import torchvision
from os.path import exists
import matplotlib.pyplot as plt

# define parameters
EPOCH = 100
BATCH_SIZE = 100
LR = 0.01
DOWNLOAD_MNIST = False
MNIST_ROOT = './../MNIST'
PKL_FILE = './pkl/test_cnn_net_params.pkl'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=5,
                padding=2
            ),  # (10, 28, 28)
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2)  # (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=3,
                padding=1
            ),  # (20, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (20, 14, 14)
            # nn.Dropout2d(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=30,
                kernel_size=3,
                padding=1
            ),  # (30, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (30, 7, 7)
        )
        self.conv1_res = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=5,
                padding=2
            ),  # (20, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4)  # (20, 7, 7)
        )
        self.l1 = nn.Sequential(
            nn.Linear(50 * 7 * 7, 100),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )
        self.l2 = nn.Sequential(
            nn.Linear(100, 10),
            nn.ReLU()
        )
        return

    def forward(self, input):
        c1 = self.conv1(input)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c1_res = self.conv1_res(c1)
        l1 = self.l1(torch.cat((torch.flatten(c3, start_dim=1), torch.flatten(c1_res, start_dim=1)),
                               dim=1))  # (30, 7, 7) + (20, 7, 7) -> (50 * 7 * 7, 1)
        l2 = self.l2(l1)
        return l2


def train(train_data, validation_data, device):
    cnn = CNN().to(device)
    if exists(PKL_FILE):
        cnn.load_state_dict(torch.load(PKL_FILE))
        print('加载旧的参数成功')
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_fun = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (image, label) in enumerate(train_data):
            # put data into device (gpu)
            image, label = image.to(device), label.to(device)
            predict = cnn(image)
            loss = loss_fun(predict, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_out = cnn(validation_data[0])
                test_predict_index = torch.max(test_out, 1)[1].data.squeeze()
                train_predict_index = torch.max(predict, 1)[1].data.squeeze()
                acc = (test_predict_index == validation_data[1]).sum().item() / validation_data[1].size(0)
                train_acc = sum(train_predict_index == label).item() / label.size(0)
                print('EPOCH:', epoch, '| STEP:', step, '| TRAIN_ACC: %.4f' % train_acc, '| VALI_ACC: %.4f' % acc,
                      '| LOSS: %.4f' % loss.item())
        torch.save(cnn.state_dict(), PKL_FILE)
        print('保存参数成功')


def showLayer(data, device):
    cnn = CNN()
    if exists(PKL_FILE):
        cnn.load_state_dict(torch.load(PKL_FILE))
        print('加载旧的参数成功')

    plt.figure('c1')
    c1 = cnn.conv1(data[0].to('cpu'))
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.imshow(c1[0][i].detach().numpy())

    plt.figure('c2')
    c2 = cnn.conv2(c1)
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(c2[0][i].detach().numpy())

    plt.figure('c3')
    c3 = cnn.conv3(c2)
    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.imshow(c3[0][i].detach().numpy())

    plt.figure('c1_res')
    c1_res = cnn.conv1_res(c1)
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.imshow(c1_res[0][i].detach().numpy())

    l1 = cnn.l1(torch.cat((torch.flatten(c3, start_dim=1), torch.flatten(c1_res, start_dim=1)),
                          dim=1))  # (30, 7, 7) + (20, 7, 7) -> (50 * 7 * 7, 1)
    out = cnn.l2(l1)
    print(torch.max(out, 1)[1].item())
    plt.show()
    pass


def main(device):
    # load mnist
    train_data = torchvision.datasets.MNIST(MNIST_ROOT, train=True, download=DOWNLOAD_MNIST,
                                            transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(MNIST_ROOT, train=False)

    # put mnist into Loader
    train_data = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data_data = (torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.).to(device)
    test_data_labels = test_data.test_labels[:2000].to(device)
    test_data = [test_data_data, test_data_labels]
    # train
    train(train_data, test_data, device)

    # showLayer
    # i = 3
    # test_data = [test_data_data[i:i+1], test_data_labels[i:i+1]]
    # showLayer(test_data, device)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)
