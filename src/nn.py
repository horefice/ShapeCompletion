import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNN(nn.Module):
    """
    A PyTorch implementation of a superclass network.
    """

    def __init__(self):
        """
        Initialize a new network.
        """
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but
        by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        print("MyNN: Forward method should be overwritten!")
        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path="../models/nn.pth"):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.pth".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.state_dict(), path)


class MyNet(MyNN):
    """
    A PyTorch implementation for shape completion
    """

    def __init__(self, n_features=80, log_transform=True):
        super().__init__()
        self.log_transform = log_transform

        # Encoder
        self.enc1 = nn.Conv3d(2, n_features, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(n_features)
        self.enc2 = nn.Conv3d(n_features, 2 * n_features, 4,
                              stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(2 * n_features)
        self.enc3 = nn.Conv3d(2 * n_features, 4 * n_features, 4,
                              stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(4 * n_features)
        self.enc4 = nn.Conv3d(4 * n_features, 8 * n_features, 4,
                              stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(8 * n_features)

        # Bottleneck
        self.fc1 = nn.Linear(8 * n_features, 8 * n_features)
        self.bn5 = nn.BatchNorm1d(8 * n_features)
        self.fc2 = nn.Linear(8 * n_features, 8 * n_features)
        self.bn6 = nn.BatchNorm1d(8 * n_features)

        # Decoder
        self.dec1 = nn.ConvTranspose3d(2 * 8 * n_features, 4 * n_features, 4,
                                       stride=1, padding=0)
        self.dbn1 = nn.BatchNorm3d(4 * n_features)
        self.dec2 = nn.ConvTranspose3d(2 * 4 * n_features, 2 * n_features, 4,
                                       stride=2, padding=1)
        self.dbn2 = nn.BatchNorm3d(2 * n_features)
        self.dec3 = nn.ConvTranspose3d(2 * 2 * n_features, n_features, 4,
                                       stride=2, padding=1)
        self.dbn3 = nn.BatchNorm3d(n_features)
        self.dec4 = nn.ConvTranspose3d(2 * n_features, 1, 4,
                                       stride=2, padding=1)

    def forward(self, x):
        enc1 = F.leaky_relu(self.bn1(self.enc1(x)))
        enc2 = F.leaky_relu(self.bn2(self.enc2(enc1)))
        enc3 = F.leaky_relu(self.bn3(self.enc3(enc2)))
        enc4 = F.leaky_relu(self.bn4(self.enc4(enc3)))
        shape = enc4.shape

        b1 = F.leaky_relu(self.bn5(self.fc1(enc4.view(enc4.size(0), -1))))
        b2 = F.leaky_relu(self.bn6(self.fc2(b1))).view(shape)

        d1 = torch.cat([b2, enc4], dim=1)
        dec1 = F.leaky_relu(self.dbn1(self.dec1(d1)))
        d2 = torch.cat([dec1, enc3], dim=1)
        dec2 = F.leaky_relu(self.dbn2(self.dec2(d2)))
        d3 = torch.cat([dec2, enc2], dim=1)
        dec3 = F.leaky_relu(self.dbn3(self.dec3(d3)))
        d4 = torch.cat([dec3, enc1], dim=1)
        dec4 = self.dec4(d4)

        out = dec4.abs()
        if self.log_transform:
            out.add_(1).log_()

        return out
