import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

wine = load_wine()

x_train, x_test, y_train, y_test = train_test_split(
	wine.data[:, :13],
	wine.target,
	test_size=0.3,
	shuffle=True)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test) 

class WineNet(torch.nn.Module):
	def __init__(self, n_neur):
		super(WineNet, self).__init__()
		self.fc1 = torch.nn.Linear(13, n_neur)
		self.ac1 = torch.nn.Sigmoid()
		self.fc2 = torch.nn.Linear(n_neur, 3)
		self.sm = torch.nn.Softmax(dim=1)
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.ac1(x)
		x = self.fc2(x)
		return x

	def inference(self, x):
		x = self.forward(x)
		x = self.sm(x)
		return x

wine_net = WineNet(5)

loss = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(wine_net.parameters(), lr=0.001)
# np.random.permutation(5)

batch_size = 10
for ep in range(1000):
	order = np.random.permutation(len(x_train))
	for start in range(0, len(x_train), batch_size):
		optim.zero_grad()
		batch_ind = order[start:start+batch_size]
		x_batch = x_train[batch_ind]
		y_batch = y_train[batch_ind]
		preds = wine_net.forward(x_batch)
		loss_val = loss(preds, y_batch)
		loss_val.backward()
		optim.step()
	if ep % 10 == 0:
		test_pred = wine_net.forward(x_test)
		test_pred = test_pred.argmax(dim=1)
		print((test_pred == y_test).float().mean())


import matplotlib.pyplot as plt
# %matplotlib inline

plt.rcParams['figure.figsize'] = (10, 8)

n_classes = 3
plot_colors = ['g', 'orange', 'black']
plot_step = 0.02

x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

xx, yy =  torch.meshgrid(torch.arange(x_min, x_max, plot_step),
                         torch.arange(y_min, y_max, plot_step))

preds = wine_net.inference(
    torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))

preds_class = preds.data.numpy().argmax(axis=1)
preds_class = preds_class.reshape(xx.shape)
plt.contourf(xx, yy, preds_class, cmap='Accent')

for i, color in zip(range(n_classes), plot_colors):
    indexes = np.where(y_train == i)
    plt.scatter(x_train[indexes, 0], 
                x_train[indexes, 1], 
                c=color, 
                label=wine.target_names[i],
                cmap='Accent')
    plt.xlabel(wine.feature_names[0])
    plt.ylabel(wine.feature_names[1])
    plt.legend()
plt.show()
