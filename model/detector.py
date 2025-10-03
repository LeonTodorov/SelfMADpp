import torch
from torch import nn

from utils.train.sam import SAM
from model.nets import TIMMModel, HRNetSeg, CLIPVision, CLIPDual, SelfMADpp

class Detector(nn.Module):
    def __init__(self, model, lr=1e-4):
        super(Detector, self).__init__()
        model_map = {
            "TIMMModel": TIMMModel,
            "HRNetSeg": HRNetSeg,
            "CLIPVision": CLIPVision,
            "CLIPDual": CLIPDual,
            "SelfMAD++": SelfMADpp,
        }
        if model in model_map:
            self.net = model_map[model](num_classes=2)
        else:
            raise ValueError(f"Unknown model type: {model}")
            
        self.cel = nn.CrossEntropyLoss()
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)
        self.model = model
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, x, target, mask):
        for i in range(2):
            pred_cls, pred_seg = self(x)
            if i == 0:
                pred_first_cls, pred_first_seg = pred_cls, pred_seg

            loss = self.cel(pred_cls, target) + self.cel(pred_seg, mask[:, 0, :, :])
            self.optimizer.zero_grad()
            loss.backward()
            
            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first_cls, pred_first_seg
    
def test():
    from utils.util_fun import load_data_config
    cfg = load_data_config("data/data_config.yaml")
    model = Detector(model="SelfMADpp", lr=1e-4)
    img = torch.randn(4, 3, cfg['image_size'], cfg['image_size']).cuda()
    label = torch.randint(0, 2, (4,)).cuda()
    mask = torch.randint(0, 2, (4, 1, cfg['image_size'], cfg['image_size'])).cuda()
    pred_cls, pred_seg = model.training_step(img, label, mask)
    print(f"Input image shape: {img.shape}")
    print(f"Predicted class shape: {pred_cls.shape}")
    print(f"Predicted segmentation shape: {pred_seg.shape}")
    