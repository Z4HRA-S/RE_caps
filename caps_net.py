# https://github.com/jindongwang/Pytorch-CapsuleNet/blob/master/capsnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# USE_CUDA = True if torch.cuda.is_available() else False
USE_CUDA = False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=24):
        super(ConvLayer, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=64,
                               kernel_size=kernel_size,
                               stride=2
                               )
        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=kernel_size,
                               stride=2
                               )
        self.conv3 = nn.Conv1d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                               )

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        return output


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=12, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()
        capsule = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=2, padding=0)
        )
        self.capsules = nn.ModuleList([
            # nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
            # kernel_size=kernel_size, stride=2, padding=0)
            capsule
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 72, -1)
        u = self.squash(u)
        return u

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=96, num_routes=32 * 72, in_channels=12, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                a_ij = a_ij.squeeze(4).mean(dim=0, keepdim=True)
                b_ij = b_ij + a_ij
        v_ij = v_j.squeeze(1)
        return v_ij

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_height=768, input_channel=3, num_caps=96):
        super(Decoder, self).__init__()
        self.num_caps = num_caps
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * num_caps, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data, th):
        classes = torch.sqrt((x ** 2).sum(2))
        # classes = F.softmax(classes, dim=0)
        # _, max_length_indices = classes.max(dim=1)
        # masked = Variable(torch.sparse.torch.eye(self.num_caps))
        masked = torch.zeros_like(classes)
        indices = classes.gt(th)
        masked[indices] = 1
        if USE_CUDA:
            masked = masked.cuda()
        # masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        masked = masked.squeeze(-1)
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_height)
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, config=None, th=0.5):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()
        self.th = th

    def forward(self, data):
        output = self.conv_layer(data)
        output = self.primary_capsules(output)
        output = self.digit_capsules(output)
        reconstructions, masked = self.decoder(output, data, self.th)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(self.th - v_c).view(batch_size, -1)
        right = F.relu(v_c - self.th).view(batch_size, -1)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        if torch.any(torch.isnan(loss)):
            print("margin_loss:",
                  f"v_c is nan:{torch.any(torch.isnan(v_c))}",
                  f"left is nan:{torch.any(torch.isnan(left))}",
                  f"right is nan:{torch.any(torch.isnan(right))}",)

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        if torch.any(torch.isnan(loss)):
            print("reconstruction_loss:",
                  f"reconstruction is nan:{torch.any(torch.isnan(reconstructions))}",
                  f"data is nan:{torch.any(torch.isnan(data))}")
        return loss * 0.0005
