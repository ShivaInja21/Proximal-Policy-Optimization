import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ConvPolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super(ConvPolicyNet, self).__init__()
        channels, height, width = obs_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_output_dim = self._compute_conv_output((channels, height, width))
        self.fc_mean = nn.Linear(conv_output_dim, act_dim)
        self.log_std_param = nn.Parameter(-0.5 * torch.ones(act_dim))

    def _compute_conv_output(self, input_shape):
        output = self.conv_layers(torch.zeros(1, *input_shape))
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv_layers(x).view(x.size(0), -1)
        mean = self.fc_mean(x)
        std_dev = torch.exp(self.log_std_param)
        return mean, std_dev

    def generate_action(self, state):
        mean, std_dev = self(state)
        dist = torch.distributions.Normal(mean, std_dev)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob.detach()

class ConvValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super(ConvValueNet, self).__init__()
        channels, height, width = obs_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_output_dim = self._compute_conv_output((channels, height, width))
        self.fc_value = nn.Linear(conv_output_dim, 1)

    def _compute_conv_output(self, input_shape):
        output = self.conv_layers(torch.zeros(1, *input_shape))
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv_layers(x).view(x.size(0), -1)
        value = self.fc_value(x)
        return value

def render_pendulum_image(theta, image_size=64):
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    center = (image_size // 2, image_size // 2)
    length = image_size // 2 - 4
    thickness = 3
    end_x = int(center[0] + length * np.sin(theta))
    end_y = int(center[1] - length * np.cos(theta))
    cv2.line(image, center, (end_x, end_y), color=255, thickness=thickness)
    return image / 255.0
