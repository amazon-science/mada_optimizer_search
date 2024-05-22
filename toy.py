import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class MNIST_FullyConnected(nn.Module):
    """
    A fully-connected NN for the MNIST task. This is Optimizable but not itself
    an optimizer.
    """
    def __init__(self, num_inp, num_hid, num_out):
        super(MNIST_FullyConnected, self).__init__()
        self.layers = nn.ModuleDict(dict(
            l1 = nn.Linear(num_inp, num_hid),
            l2 = nn.Linear(num_hid, num_out),
            l3 = nn.Linear(num_out, num_out)
        ))
        self.layer1 = nn.Linear(num_inp, num_hid)
        self.layer2 = nn.Linear(num_hid, num_out)

    def initialize(self):
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.layer2.weight, a=math.sqrt(5))

    def forward(self, x):
        """Compute a prediction."""
        x = self.layers.l1(x)
        x = torch.tanh(x)
        x = self.layers.l2(x)
        x = torch.tanh(x)
        x = self.layers.l3(x)
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

BATCH_SIZE = 256
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dl_train = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
dl_test = torch.utils.data.DataLoader(mnist_test, batch_size=10000, shuffle=False)

model = MNIST_FullyConnected(28 * 28, 128, 10).to(DEVICE)


import gdtuo

optim = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))

mw = gdtuo.ModuleWrapper(model, optimizer=optim)
mw.initialize()
torch.manual_seed('1234')
for i in range(1, EPOCHS+1):
    running_loss = 0.0
    for j, (features_, labels_) in enumerate(dl_train):
        mw.begin() # call this before each step, enables gradient tracking on desired params
        features, labels = torch.reshape(features_, (-1, 28 * 28)).to(DEVICE), labels_.to(DEVICE)
        pred = mw.forward(features)
        loss = F.nll_loss(pred, labels)
        mw.zero_grad()
        loss.backward(create_graph=True) # important! use create_graph=True
        mw.step()
        running_loss += loss.item() * features_.size(0)
    train_loss = running_loss / len(dl_train.dataset)
    print("EPOCH: {}, TRAIN LOSS: {}".format(i, train_loss))