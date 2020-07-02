class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 0.75 * input_dim)
        self.fc2 = nn.Linear(0.75 * input_dim, 0.5 * input_dim)
        self.fc3 = nn.Linear(0.5 * input_dim, 0.33 * input_dim)
        self.fc4 = nn.Linear(0.33 * input_dim, 0.25 * input_dim)
        self.fc5 = nn.Linear(0.25 * input_dim, 0.33 * input_dim)
        self.fc6 = nn.Linear(0.33 * input_dim, 0.5 * input_dim)
        self.fc7 = nn.Linear(0.5 * input_dim, 0.75 * input_dim)
        self.fc8 = nn.Linear(0.75 * input_dim, input_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = self.fc8(x)
