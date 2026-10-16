"""GigaTIME-Flash architecture, adapted from the official testing notebook.

Upstream: prov-gigatime/GigaTIME, commit
acc49b2567fd15ce95b5a82caaab5f0a295b88f7,
scripts/gigatime_flash_testing.ipynb. Apache-2.0; see THIRD_PARTY_LICENSE.
Only the architecture is included; no upstream download or inference cells run.
"""
import torch
from torch import nn
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import _create_vision_transformer

class LoraConfig:
    def __init__(self, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or []


class LoRALinear(nn.Module):
    def __init__(self, base_layer, config):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = {"default": config.lora_alpha / config.r}
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(config.lora_dropout)})
        self.lora_A = nn.ModuleDict({"default": nn.Linear(base_layer.in_features, config.r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(config.r, base_layer.out_features, bias=False)})

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        result = self.base_layer(x)
        update = self.lora_B["default"](self.lora_A["default"](self.lora_dropout["default"](x)))
        return result + update * self.scaling["default"]


class BaseModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class LoRAConv2d(nn.Module):
    """PEFT-compatible patch projection omitted by the notebook fallback.

    The released checkpoint includes these adapters. Keeping only the base
    convolution would silently discard 9,216 learned parameters.
    """
    def __init__(self, base_layer, config):
        super().__init__()
        self.base_layer = base_layer
        self.scaling = config.lora_alpha / config.r
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(config.lora_dropout)})
        self.lora_A = nn.ModuleDict({"default": nn.Conv2d(
            base_layer.in_channels, config.r, base_layer.kernel_size,
            stride=base_layer.stride, padding=base_layer.padding,
            dilation=base_layer.dilation, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Conv2d(
            config.r, base_layer.out_channels, 1, bias=False)})

    def forward(self, x):
        return self.base_layer(x) + self.scaling * self.lora_B["default"](
            self.lora_A["default"](self.lora_dropout["default"](x)))


class PeftFallbackModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = BaseModelWrapper(model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model.model, name)

    def forward(self, *args, **kwargs):
        return self.base_model.model(*args, **kwargs)


def apply_lora(module, config):
    for child_name, child in list(module.named_children()):
        if child_name in config.target_modules and isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, config))
        elif child_name in config.target_modules and isinstance(child, nn.Conv2d):
            setattr(module, child_name, LoRAConv2d(child, config))
        else:
            apply_lora(child, config)


def make_vit_small_patch16_dinov2():
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, init_values=1e-5,
        mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, img_size=224,
    )
    return _create_vision_transformer("vit_small_patch14_dinov2", pretrained=False, **model_args)

class GigaTIMEFlash(nn.Module):
    def __init__(self, num_classes=23):
        super().__init__()
        vit_encoder = make_vit_small_patch16_dinov2()
        config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "proj"])
        apply_lora(vit_encoder, config)
        self.encoder = PeftFallbackModel(vit_encoder)
        self.num_classes = num_classes

        self.decoder4 = self._decoder_block(384, 192)
        self.decoder3 = self._decoder_block(192, 96)
        self.decoder2 = self._decoder_block(96, 48)
        self.decoder1 = self._decoder_block(48, 24)

        self.skip1 = self._skip_block(384, 48, 3)
        self.skip2 = self._skip_block(384, 96, 2)
        self.skip3 = self._skip_block(384, 192, 1)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

        self.encoder.patch_embed.img_size = (256, 256)
        self.encoder.patch_embed.grid_size = (16, 16)
        self.encoder.patch_embed.num_patches = 256

    @staticmethod
    def _decoder_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    @staticmethod
    def _skip_block(in_channels, out_channels, times):
        layers = []
        for _ in range(times):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _tokens_to_patch(x):
        batch_size, num_tokens, channels = x.shape
        grid = int(num_tokens ** 0.5)
        if grid * grid != num_tokens:
            raise ValueError(f"Expected square patch tokens, got {num_tokens}")
        return x.permute(0, 2, 1).contiguous().view(batch_size, channels, grid, grid)

    @staticmethod
    def _resize_pos_embed(pos_embed, new_grid_size):
        cls_tok, grid_tok = pos_embed[:, :1], pos_embed[:, 1:]
        old_num = int(grid_tok.shape[1] ** 0.5)
        grid_tok = grid_tok.reshape(1, old_num, old_num, -1).permute(0, 3, 1, 2)
        grid_tok = torch.nn.functional.interpolate(grid_tok, size=new_grid_size, mode="bicubic", align_corners=False)
        grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(1, new_grid_size[0] * new_grid_size[1], -1)
        return torch.cat((cls_tok, grid_tok), dim=1)

    def forward(self, x):
        enc_outs = []
        x = self.encoder.patch_embed(x)
        _, num_tokens, _ = x.shape
        grid = int(num_tokens ** 0.5)

        pos_emb = self._resize_pos_embed(self.encoder.pos_embed, (grid, grid))
        x = x + pos_emb[:, 1:]
        x = self.encoder.patch_drop(x)
        x = self.encoder.norm_pre(x)

        for idx, block in enumerate(self.encoder.blocks):
            x = block(x)
            if idx in {3, 5, 8, 11}:
                enc_outs.append(x)

        x = self.decoder4(self._tokens_to_patch(enc_outs[3]))
        x = self.decoder3(x + self.skip3(self._tokens_to_patch(enc_outs[2])))
        x = self.decoder2(x + self.skip2(self._tokens_to_patch(enc_outs[1])))
        x = self.decoder1(x + self.skip1(self._tokens_to_patch(enc_outs[0])))
        x = self.final_conv(x)

        if x.shape[-1] != 256:
            x = nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)
        return x
