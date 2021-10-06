## Design Basic DQN using Pytorch
## Experience Replay
## Clip Rewards
## Train for ieee-2 and ieee-4
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ieee4_net(nn.Module):
    def __init__(self, in_channels=4, num_actions=200):
        super(ieee4_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1,padding =1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


model = ieee4_net()
device = torch.device("cpu")
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print("Model's state_dict:")




for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])



class ieee2_net(nn.Module):
    def __init__(self, input, num_actions,p=0.5):
        super(ieee2_net, self).__init__()
        #self.conv1 = nn.Conv2d(inputChannels, 8, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(input, 16 * 3 * 2);
        # self.fc2 = nn.Linear(16 * 3 * 2, 16 * 3 * 2 * 2);
        # self.fc3 = nn.Linear(16 * 3 * 2 * 2, num_actions);
        self.fc1 = nn.Linear(input, input*2);
        self.fc2 = nn.Linear(input*2, input*2*2);
        self.fc3 = nn.Linear(input*2*2, num_actions);
        #self.fc3 = nn.Linear(18, 54);
        #self.fc4 = nn.Linear(54, num_actions)
        self.drop_layer1 = nn.Dropout(p=p)
        self.drop_layer2 = nn.Dropout(p=p)

    def forward(self, x):
        #x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.drop_layer(x);

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.drop_layer1(x);
        x=F.relu(self.fc2(x))
        x = self.drop_layer2(x);
        x = F.relu(self.fc3(x))
        return x

        # x = F.relu(self.fc1(x));
        # x = F.relu(self.fc2(x));
        # x=self.drop_layer(x);
        # x = F.relu(self.fc3(x));
        # return self.fc4(x);

