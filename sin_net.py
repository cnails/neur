import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

x_train = torch.rand(2000)
x_train = x_train * 20.0 - 10.0

y_train = torch.sin(x_train)
# plt.plot(x_train.numpy(), y_train.numpy(), '+')
# plt.title('test')
noise = torch.randn(y_train.shape) / 2.0
# plt.plot(x_train.numpy(), noise.numpy(), 'o')
# plt.axis([-10, 10, -1, 1])
# plt.show()
y_train = y_train + noise
# plt.plot(x_train.numpy(), y_train.numpy(), 'o')
# plt.show()
# exit()
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)
x_valid = torch.linspace(-15, 15, 100)
y_valid = torch.sin(x_valid.data)
x_valid.unsqueeze_(1)
y_valid.unsqueeze_(1)

class Sine(torch.nn.Module):
	def __init__(self, n_neur):
		super(Sine, self).__init__()
		self.fc1 = torch.nn.Linear(1, n_neur)
		self.act1 = torch.nn.Sigmoid()
		self.fc2 = torch.nn.Linear(n_neur, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act1(x)
		x = self.fc2(x)
		return x

sine = Sine(50)

def pred(net, x, y):
	y_pred = net.forward(x)

	plt.plot(x.numpy(), y.numpy(), 'o', label='dataset')
	plt.plot(x.numpy(), y_pred.data.numpy(), 'o', label='train')
	plt.legend(loc='upper right')

# pred(sine, x_valid, y_valid)
# plt.show()
optim = torch.optim.Adam(sine.parameters(), lr=0.01)

def loss(pred, targ):
	sq = (pred - targ) * (pred - targ)
	return sq.mean()

for epoch in range(8000):
	optim.zero_grad()

	y_pred = sine.forward(x_train)
	loss_val = loss(y_pred, y_train)
	loss_val.backward()

	optim.step()

y_pred = sine.forward(x_valid)

pred(sine, x_valid, y_valid)
plt.show()
