import torch
from torch import nn
from torchsummary import summary


class ConvNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64 * 8 * 8, num_classes)
        
        self.pool = nn.MaxPool2d(2)
        
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = ConvNet()
    model.load_state_dict(torch.load("conv_net.pth", map_location="cpu", weights_only=True))
    summary(model, input_size=(3, 64, 64))
    # data_statistics = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}