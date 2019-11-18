import torch
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True

import torchvision.datasets
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

x_train = MNIST_train.data
y_train = MNIST_train.targets
x_test = MNIST_test.data
y_test = MNIST_test.targets

x_train = x_train.float()
x_test = x_test.float()

import matplotlib.pyplot as plt
# plt.imshow(x_test[0, :, :])
# plt.show()
print(y_test[0])

x_train = x_train.reshape([-1, 28 * 28])
x_test = x_test.reshape([-1, 28 * 28])

class Net(torch.nn.Module):
  def __init__(self, n_neur):
    super(Net, self).__init__()
    self.fc1 = torch.nn.Linear(28 * 28, n_neur)
    self.ac1 = torch.nn.Sigmoid()
    self.fc2 = torch.nn.Linear(n_neur, 10)

  def forward(self, x):
    x = self.fc1(x)
    x = self.ac1(x)
    x = self.fc2(x)
    return x

net = Net(200)

# device = torch.device('cuda:0')
net = net #.to(device)

loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.0001)

batch_size = 100

test_accuracy = []
train_accuracy = []
test_loss = []
train_loss = []

x_test = x_test #.to(device)
y_test = y_test #.to(device)

for epoch in range(3):
  order = np.random.permutation(len(x_train))
#   print(order)
#   exit()
  for start in range(0, len(x_train), batch_size):
    optim.zero_grad()
    batch_ind = order[start:start+batch_size]
    x_batch = x_train[batch_ind] #.to(device)
    y_batch = y_train[batch_ind] #.to(device)

    preds = net.forward(x_batch)

    loss_val = loss(preds, y_batch)
    loss_val.backward()

    optim.step()

  train_preds = net.forward(x_batch)
  train_loss.append(loss(train_preds, y_batch))

  test_preds = net.forward(x_test)
  test_loss.append(loss(test_preds, y_test))


  tr_accur = (train_preds.argmax(dim=1) == y_batch).float().mean()
  accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
  test_accuracy.append(accuracy)
  train_accuracy.append(tr_accur)
  print(f'test = {accuracy}')
  print(f'train = {tr_accur}')

lol = (net.forward(x_test)).argmax(dim=1).float().mean()
print(lol)

plt.plot(test_accuracy)
plt.plot(train_accuracy)

plt.plot(test_loss)
plt.plot(train_loss)
