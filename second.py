import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)


class RegressionNet(torch.nn.Module):
    def __init__(self, n_neur):
        super(RegressionNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_neur)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_neur, n_neur)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_neur, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

net = RegressionNet(50)

def target_function(x):
    return 2**x * torch.sin(2**-x)

# ------Dataset preparation start--------:
x_train =  torch.linspace(-10, 5, 1000)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 1000)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Dataset preparation end--------:

def metric(pred, target):
    return (pred - target).abs().mean()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

def loss(pred, targ):
	sq = (pred - targ) * (pred - targ)
	return sq.mean()
    # return ((torch.abs(pred - target)).mean())

for epoch_index in range(20000):
    optimizer.zero_grad()

    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)
    loss_value.backward()
    optimizer.step()
    
def predict(net, x, y):
    y_pred = net.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    plt.plot(x.numpy(), y_pred.data.numpy(), 'o', c='r', label='Prediction');
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')


predict(net, x_validation, y_validation)

plt.show()

print(metric(net.forward(x_validation), y_validation).item())