# https://github.com/jindongwang/Pytorch-CapsuleNet/blob/master/capsnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

conv = {
    "in_channel": 5,
    "out_channel": 128
}
primary_caps = {
    "in_channel": conv["out_channel"],
    "out_channel": 8,
    "num_caps": 20
}

digit_caps = {
    "in_channel": primary_caps["num_caps"],
    "out_channel": 16,
    "num_caps": 96,
    "num_route": 8 * 24
}


def squash(input_tensor):
    original_size = input_tensor.size()
    input_tensor = input_tensor.squeeze()
    norm = torch.linalg.norm(input_tensor, dim=-1).unsqueeze(dim=-1)
    output = (norm ** 2) / (1 + (norm ** 2)) * (input_tensor / norm)
    output = output.view(original_size)
    return output


class ConvLayer(nn.Module):
    def __init__(self, in_channels=conv["in_channel"], out_channels=conv["out_channel"], kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=64,
                               kernel_size=kernel_size,
                               stride=3
                               )
        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=128,
                               kernel_size=kernel_size,
                               stride=3
                               )

    def forward(self, x):
        output = F.leaky_relu(self.conv1(x))
        output = F.leaky_relu(self.conv2(output))
        return output


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=primary_caps["num_caps"],
                 in_channels=primary_caps["in_channel"],
                 out_channels=primary_caps["out_channel"], kernel_size=12):
        super(PrimaryCaps, self).__init__()

        capsule = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=3, padding=0)
        )
        self.capsules = nn.ModuleList([
            capsule
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        num_route = u[0].size(-1) * u[0].size(-2)
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), num_route, -1)
        u = squash(u)
        return u


class DigitCaps(nn.Module):
    def __init__(self, device, num_capsules=digit_caps["num_caps"],
                 num_routes=digit_caps["num_route"],
                 in_channels=digit_caps["in_channel"],
                 out_channels=digit_caps["out_channel"]):

        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.W = nn.Parameter(torch.randn(num_routes, num_capsules, out_channels, in_channels))
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        # x = x.transpose(-1, -2)
        # x = torch.stack([x] * self.num_capsules, dim=1).unsqueeze(-1)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        u_hat = torch.stack(
            [torch.matmul(self.W, x[i])
             for i in range(batch_size)])

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if "cuda" in self.device.type:
            u_hat = u_hat.cuda()
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.concat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.concat([v_j] * self.num_routes, dim=1))
                a_ij = a_ij.squeeze(4).mean(dim=0, keepdim=True)
                b_ij = b_ij + a_ij
        v_ij = v_j.squeeze(1)
        return v_ij


class CapsNet(nn.Module):
    def __init__(self, device, num_class=96):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps(device=device)

    def forward(self, data):
        output = self.conv_layer(data)
        output = self.primary_capsules(output)
        output = self.digit_capsules(output)
        return output

    def margin_loss(self, x, labels):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.5 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.5).view(batch_size, -1)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()
        return loss
