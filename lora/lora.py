import torch
from .loralib.layers import PlainMultiheadAttentionLoRA

INDEX_POSITIONS_TEXT = {
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-L/14': {'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
    'ViT-L/14@336px': {'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def apply_lora_clip(model, training_type, model_name, target_modules, lora_rank, lora_alpha, lora_dropout, device, position="all"):
    list_lora_layers = []
    
    if training_type in ["image_encoder_only", "text_image_contrastive"]:
        for i, block in enumerate(model.visual.transformer.resblocks):
            if i in INDEX_POSITIONS_VISION[model_name][position]:
                for name, submodule in block.named_children():
                    if isinstance(submodule, torch.nn.MultiheadAttention):
                        pass
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=target_modules, r=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout).to(device)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    if training_type == "text_image_contrastive":
        for i, block in enumerate(model.transformer.resblocks):
            if i in INDEX_POSITIONS_TEXT[position]:
                for name, submodule in block.named_children():
                    if isinstance(submodule, torch.nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule, enable_lora=target_modules, r=lora_rank, lora_alpha=lora_alpha, dropout_rate=lora_dropout).to(device)
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

    return list_lora_layers
