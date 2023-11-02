

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    g = torch.Generator()
    g.manual_seed(seed)
    
def ssl_encode(encoder, image):

    encoder.eval()

    embedding = encoder(image_one=image, image_two=None)
    return embedding[0].detach()