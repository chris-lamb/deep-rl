import torch


class DQN(torch.nn.Module):
    '''Deep Q Learning convolutionl neural network.

    Convolutional neural network class for estimating Q function. Comprised of
    two convolutional layers, one fully connected hidden layer, and a Linear
    output for each possible action.

    Architecture from Mihn et al. https://arxiv.org/abs/1312.5602

    '''

    def __init__(self, in_channels=4, num_actions=2):
        super(DQN, self).__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.features = self._init_features()
        self.classifier = self._init_classifier()

    def _init_features(self):
        layers = []
        # 80 x 80 x 4 initial dimensions
        layers.append(torch.nn.Conv2d(self.in_channels, 16, kernel_size=8, stride=4, padding=2))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # 20 x 20 x 16 feature maps
        layers.append(torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 10 x 10 x 32 feature maps
        return torch.nn.Sequential(*layers)

    def _init_classifier(self):
        layers = []
        layers.append(torch.nn.Linear(10*10*32, 256))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(256, self.num_actions))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Policy(torch.nn.Module):
    '''Actor Critic convolutionl neural network.

    Convolutional neural network class for estimating action head and value
    head function. Comprised of three convolutional layers and a linear
    output for each possible action as well as a single linear output for value.

    '''

    def __init__(self, in_channels, num_actions):
        super(Policy, self).__init__()
        self.temperature = 1.0
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.conv1 = torch.nn.Conv2d(in_channels, 16, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.features = self._init_features()
        self.action_head = self._init_action_head()
        self.value_head = self._init_value_head()

        self.saved_actions = []
        self.rewards = []

    def _init_features(self):
        layers = []
        # 80 x 80 x in_channels initial dimensions 3D array
        layers.append(self.conv1)
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # 40 x 40 x 32 feature maps
        layers.append(self.conv2)
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 20 x 20 x 32
        layers.append(self.conv3)
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 10 x 10 x 32 feature maps
        return torch.nn.Sequential(*layers)

    def _init_action_head(self):
        return torch.nn.Linear(32*10*10, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(32*10*10, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        action = torch.nn.functional.softmax(self.action_head(x) / self.temperature, dim=-1)
        value = self.value_head(x)
        return action, value


class LSTMPolicy(torch.nn.Module):
    '''Actor Critic convolutionl neural network with LSTM.

    Convolutional neural network class for estimating action head and value
    head function. Comprised of four convolutional layers, an LSTM layer with
    100 hidden nodes, and a linear output for each possible action as well as
    a single linear output for value.

    '''

    def __init__(self, in_channels, num_actions):
        super(LSTMPolicy, self).__init__()
        self.temperature = 1.0
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.conv1 = torch.nn.Conv2d(in_channels, 16, 4, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.features = self._init_features()
        self.lstm = self._init_lstm()
        self.action_head = self._init_action_head()
        self.value_head = self._init_value_head()

        self.saved_actions = []
        self.rewards = []

    def _init_features(self):
        layers = []
        # 80 x 80 x in_channels initial dimensions 3D array
        layers.append(self.conv1)
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # 40 x 40 x 32 feature maps
        layers.append(self.conv2)
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 20 x 20 x 32
        layers.append(self.conv3)
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 10 x 10 x 32 feature maps
        layers.append(self.conv4)
        layers.append(torch.nn.BatchNorm2d(32))
        layers.append(torch.nn.ReLU(inplace=True))
        # 5 x 5 x 32 feature maps
        return torch.nn.Sequential(*layers)

    def _init_lstm(self):
        return torch.nn.LSTMCell(5*5*32, 100)

    def _init_action_head(self):
        return torch.nn.Linear(100, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(100, 1)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = self.features(x)

        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        action = torch.nn.functional.softmax(self.action_head(x) / self.temperature, dim=-1)
        value = self.value_head(x)
        return action, value, (hx, cx)


MODELS = {'dqn': DQN,
          'a2c': Policy,
          'a2c-lstm': LSTMPolicy}
