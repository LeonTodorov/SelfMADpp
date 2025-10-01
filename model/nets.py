import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.segformer.decoder import SegformerDecoder
from segmentation_models_pytorch.base.heads import SegmentationHead
import torch.nn.functional as F
import open_clip

from lora.lora import apply_lora_clip
from lora.loralib.utils import mark_only_lora_as_trainable

@torch.jit.interface
class ModuleInterface(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

class TIMMModel(nn.Module):
    def __init__(self, num_classes):
        super(TIMMModel, self).__init__()
        self.model = timm.create_model("hrnet_w18", pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

class HRNetSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cls_net = timm.create_model("hrnet_w18", pretrained=True, num_classes=num_classes)

        self.decoder = SegformerDecoder(
            encoder_channels=(3, 64, 128, 256, 512, 1024),
            encoder_depth=5,
            segmentation_channels=64,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            upsampling=4
        )
    def forward_features(self, x):
        out_seg_features = [x]
        x = self.cls_net.conv1(x)
        x = self.cls_net.bn1(x)
        x = self.cls_net.act1(x)
        out_seg_features.append(x)
        
        x  = self.cls_net.conv2(x)
        x = self.cls_net.bn2(x)
        x = self.cls_net.act2(x)
        yl = self.cls_net.stages(x)
        incre_modules_outs = [incre(x) for incre, x in zip(self.cls_net.incre_modules, yl)]
        out_seg_features.extend(incre_modules_outs)

        y = incre_modules_outs[0]
        for i in range(1, len(incre_modules_outs)):
            down: ModuleInterface = self.cls_net.downsamp_modules[i - 1]  # needed for torchscript module indexing
            y = incre_modules_outs[i] + down.forward(y)

        y = self.cls_net.final_layer(y)
        return y, out_seg_features
    
    def forward_head(self, x, pre_logits: bool = False):
        x = self.cls_net.global_pool(x)
        x = self.cls_net.head_drop(x)
        return x if pre_logits else self.cls_net.classifier(x)

    def forward(self, x):
        out_clf, out_seg_features = self.forward_features(x)
        out_clf = self.forward_head(out_clf)
        out_seg_features = self.decoder(out_seg_features)
        out_seg_features = self.segmentation_head(out_seg_features)
        return out_clf, out_seg_features

class CLIPVision(nn.Module):
    def __init__(self, num_classes):
        super(CLIPVision, self).__init__()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-L-14-336-quickgelu", pretrained="openai")
        apply_lora_clip(
            model=self.clip_model,
            training_type="image_encoder_only",
            model_name="ViT-L/14@336px",
            target_modules=["q", "v"],
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            device="cuda",
            position="all"
        )
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_preprocess.transforms = [self.clip_preprocess.transforms[0], self.clip_preprocess.transforms[-1]]
        self.classifier = nn.Linear(768, bias=True, out_features=num_classes)
        
    def forward(self, x):
        out_clip = self.clip_model.encode_image(self.clip_preprocess(x))
        out_clf = self.classifier(out_clip)
        return out_clf

class CLIPDual(nn.Module):
    def __init__(self, **kwargs):
        super(CLIPDual, self).__init__()
        self.class_names = ["bona-fide presentation", "face image morphing attack"]
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-336-quickgelu", pretrained="openai"
        )
        self.clip_preprocess.transforms = [
            self.clip_preprocess.transforms[0], # Resize
            self.clip_preprocess.transforms[-1] # Normalize
        ]
        apply_lora_clip(
            model=self.clip_model,
            training_type="text_image_contrastive",
            model_name="ViT-L/14@336px",
            target_modules=["q", "v"],
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            device="cuda",
            position="all"
        )
        mark_only_lora_as_trainable(self.clip_model)
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14-336-quickgelu")
        
    def forward(self, x):
        x = self.clip_preprocess(x)
        image_features = self.clip_model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_tokens = self.tokenizer(self.class_names).to(x.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features * self.clip_model.logit_scale.exp() @ text_features.T
        return logits

class SelfMADpp(nn.Module):
    def __init__(self, num_classes):
        super(SelfMADpp, self).__init__()
        if torch.cuda.device_count() == 1:
            self.device_main = torch.device("cuda")
            self.device_clip = torch.device("cuda")
        else:
            self.device_main = torch.device("cuda:0")
            self.device_clip = torch.device("cuda:1")

        # ─── HRNet ───
        self.cls_net = timm.create_model(
            "hrnet_w18", pretrained=True, num_classes=num_classes
        ).to(self.device_main)
        self.decoder = SegformerDecoder(
            encoder_channels=(3, 64, 128, 256, 512, 1024),
            encoder_depth=5,
            segmentation_channels=256,
        ).to(self.device_main)
        self.segmentation_head = SegmentationHead(
            in_channels=256, out_channels=num_classes, kernel_size=3, upsampling=4
        ).to(self.device_main)

        # ─── CLIP ───
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-336-quickgelu", pretrained="openai"
        )
        apply_lora_clip(
            model=self.clip_model,
            training_type="text_image_contrastive",
            model_name="ViT-L/14@336px",
            target_modules=["q", "v"],
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            device=self.device_clip,
            position="all",
        )
        mark_only_lora_as_trainable(self.clip_model)
        self.clip_model = self.clip_model.to(self.device_clip)
        self.clip_preprocess.transforms = [
            self.clip_preprocess.transforms[0], # Resize
            self.clip_preprocess.transforms[-1], # Normalize
        ]

        # ─── CLIP text ───
        self.class_names = ["bona-fide presentation", "face image morphing attack"]
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14-336-quickgelu")

        # ─── Fusion ───
        self.clf_lin  = nn.Linear(4096, 512, bias=True).to(self.device_main)
        self.clip_lin = nn.Linear(768, 512, bias=True).to(self.device_main)
        self.gate_fc  = nn.Linear(512 * 2, 512).to(self.device_main)
        self.final_classifier = nn.Linear(512, num_classes).to(self.device_main)

    def forward_features(self, x):
        seg_feats = [x]
        x = self.cls_net.conv1(x); x = self.cls_net.bn1(x); x = self.cls_net.act1(x)
        seg_feats.append(x)
        x = self.cls_net.conv2(x); x = self.cls_net.bn2(x); x = self.cls_net.act2(x)
        xs = self.cls_net.stages(x)
        incr = [inc(xi) for inc, xi in zip(self.cls_net.incre_modules, xs)]
        seg_feats.extend(incr)
        y = incr[0]
        for i in range(1, len(incr)):
            y = incr[i] + self.cls_net.downsamp_modules[i - 1](y)
        y = self.cls_net.final_layer(y)
        return y, seg_feats

    @staticmethod
    def region_pooling(hr_map: torch.Tensor, seg_maps: torch.Tensor, eps=1e-6):
        B, C, h, w = hr_map.shape
        seg_probs = torch.softmax(seg_maps, dim=1)[:, 1:2]  # [B,1,H,W]
        seg_probs_resized = F.interpolate(seg_probs, size=(h, w), mode="bilinear", align_corners=False)

        fg_mask = seg_probs_resized
        bg_mask = 1.0 - seg_probs_resized

        fg_mask_sum = fg_mask.sum(dim=[2, 3], keepdim=True).clamp(min=eps)
        bg_mask_sum = bg_mask.sum(dim=[2, 3], keepdim=True).clamp(min=eps)

        fg_mask = fg_mask / fg_mask_sum
        bg_mask = bg_mask / bg_mask_sum

        fg_feat = (hr_map * fg_mask).sum(dim=[2, 3])
        bg_feat = (hr_map * bg_mask).sum(dim=[2, 3])

        return torch.cat([fg_feat, bg_feat], dim=1)

    def forward(self, x):
        x_main = x.to(self.device_main)
        x_clip = x.to(self.device_clip)

        # ─── CLIP vision ───
        clip_input = self.clip_preprocess(x_clip)
        out_clip = self.clip_model.encode_image(clip_input)
        out_clip = F.normalize(out_clip, dim=-1).to(self.device_main)

        # ─── CLIP text ───
        text_tokens = self.tokenizer(self.class_names).to(x_clip.device)
        text_feats = self.clip_model.encode_text(text_tokens)
        text_feats = F.normalize(text_feats, dim=-1).to(self.device_main)
        logit_scale = self.clip_model.logit_scale.exp().to(self.device_main)
        logits_clip = out_clip @ text_feats.T * logit_scale

        # ─── HRNet ───
        hr_map, enc_feats = self.forward_features(x_main)
        dec_out = self.decoder(enc_feats)
        seg_maps = self.segmentation_head(dec_out)
        hr_vec = self.region_pooling(hr_map, seg_maps)
        hr_vec = F.normalize(hr_vec, dim=-1)

        # ─── Fusion ───
        out_clip_512 = self.clip_lin(out_clip)
        hr_vec_512 = self.clf_lin(hr_vec)
        concat = torch.cat([hr_vec_512, out_clip_512], dim=-1)
        gates = torch.sigmoid(self.gate_fc(concat))
        fused = gates * hr_vec_512 + (1 - gates) * out_clip_512
        logits_fusion = self.final_classifier(fused)
        logits_final = (logits_fusion + logits_clip) / 2.0

        return logits_final, seg_maps
    
def test():
    net = SelfMADpp(num_classes=2)
    print(net)