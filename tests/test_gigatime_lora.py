"""Check adapter arithmetic independently of learned checkpoints."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")


def test_patch_conv_adapter_matches_merged_convolution_in_eval_mode():
    from experiments.gigatime_flash.model import LoRAConv2d, LoraConfig
    torch.manual_seed(17)
    base = torch.nn.Conv2d(3, 7, 3, stride=2, padding=1)
    module = LoRAConv2d(base, LoraConfig(r=2, lora_alpha=4)).eval()
    x = torch.randn(2, 3, 17, 19)
    delta = torch.einsum("or,rihw->oihw", module.lora_B["default"].weight[:, :, 0, 0],
                         module.lora_A["default"].weight)
    expected = torch.nn.functional.conv2d(x, base.weight+2*delta, base.bias, stride=2, padding=1)
    torch.testing.assert_close(module(x), expected, rtol=1e-5, atol=1e-6)


def test_marker_cache_invalidates_when_original_rgb_changes(tmp_path):
    import numpy as np
    from experiments.gigatime_flash.run_pilot import predict_maps
    from bora.virtual_markers import sampling_plan

    class Model:
        calls = 0

        def to(self, device):
            return self

        def __call__(self, tensor):
            self.calls += 1
            return torch.zeros((tensor.shape[0], 23, 256, 256))

    model = Model()
    rgb = np.zeros((8, 8, 3), np.uint8)
    plan = sampling_plan(rgb.shape[:2], [.25, .25], .25)
    config = {"normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
              "channels": [str(i) for i in range(23)]}
    identity = {"checkpoint_sha256": "test", "model_family": "cnn"}
    predict_maps(rgb, model, plan, config, tmp_path, 1, "cpu", identity)
    predict_maps(rgb, model, plan, config, tmp_path, 1, "cpu", identity)
    assert model.calls == 1
    rgb[0, 0] = 255
    predict_maps(rgb, model, plan, config, tmp_path, 1, "cpu", identity)
    assert model.calls == 2
