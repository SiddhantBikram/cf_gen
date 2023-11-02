# import torch
# import timm
from config import *
from helper import *

class Uncertainty(torch.nn.Module):

    def __init__(self):
        super(Uncertainty, self).__init__()

        self.classifiers = []

        for _ in range(config.num_classifier):
            self.classifiers.append(utils.get_classifier(config.classifier, config.f_dim, config.output_dim, config.linear_bias).to(config.device))

        def forward(self, x):
            x = self.encoder(x)

            return x

