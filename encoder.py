import torch
import timm

class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        
        self.encoder = timm.create_model('resnet101', pretrained=True).to(device)

        def forward(self, x):
            x = self.encoder(x)

            return x
