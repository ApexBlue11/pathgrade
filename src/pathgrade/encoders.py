"""Patch encoders, with commercial licensing enforced in code.

Every entry records its licence. Encoders that forbid commercial use - or, in
Prov-GigaPath's case, forbid *any* deployment - are present but refuse to load
unless the caller explicitly opts in. This is deliberate: the licence question
is the single thing most likely to sink a pathology product, and a comment in a
README does not stop anyone from importing the wrong model at 2am.

Two traps worth knowing about, both encoded below:

* Virchow **v1** is Apache-2.0 but Virchow**2** is CC-BY-NC-ND-4.0.
* H-optimus-**0** is Apache-2.0 but H-optimus-**1** is CC-BY-NC-ND-4.0.

In both cases the newer, stronger model is the restricted one. Reaching for the
latest version by reflex is exactly how a project ends up unshippable.

Prov-GigaPath deserves its own note: Hugging Face tags it ``apache-2.0``, which
covers the *code*, but the model card's intended-use section states that any
deployed use case, commercial or otherwise, is out of scope. The metadata tag
alone is not diligence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

Pooling = Literal["cls", "cls_mean"]


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_hub_id: str
    embed_dim: int                 # width of the vector this produces per patch
    licence: str
    commercial_ok: bool
    patch_px: int = 224
    target_mpp: float = 0.5        # 20x equivalent
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    pooling: Pooling = "cls"
    timm_kwargs: dict = field(default_factory=dict)
    notes: str = ""


REGISTRY: dict[str, EncoderSpec] = {
    # ---------------- commercially usable ----------------
    "h-optimus-0": EncoderSpec(
        name="h-optimus-0",
        hf_hub_id="hf-hub:bioptimus/H-optimus-0",
        embed_dim=1536,
        licence="Apache-2.0",
        commercial_ok=True,
        mean=(0.707223, 0.578729, 0.703617),
        std=(0.211883, 0.230117, 0.177517),
        pooling="cls",
        timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
        notes="1.1B ViT-g/14, 500k+ WSIs. Default choice: strongest Apache-2.0 encoder.",
    ),
    "virchow": EncoderSpec(
        name="virchow",
        hf_hub_id="hf-hub:paige-ai/Virchow",
        embed_dim=2560,            # 1280 CLS + 1280 mean patch tokens
        licence="Apache-2.0",
        commercial_ok=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        pooling="cls_mean",
        timm_kwargs={"mlp_layer": None, "act_layer": None},
        notes="v1 only. Virchow2 is CC-BY-NC-ND and must not be substituted.",
    ),
    "hibou-l": EncoderSpec(
        name="hibou-l",
        hf_hub_id="hf-hub:histai/hibou-L",
        embed_dim=1024,
        licence="Apache-2.0",
        commercial_ok=True,
        mean=(0.7068, 0.5755, 0.7220),
        std=(0.1950, 0.2316, 0.1816),
        pooling="cls",
        notes="Lighter fallback if H-optimus-0 inference cost is a problem.",
    ),
    "midnight": EncoderSpec(
        name="midnight",
        hf_hub_id="hf-hub:kaiko-ai/midnight",
        embed_dim=3072,            # 1536 CLS + 1536 mean patch tokens
        licence="MIT",
        commercial_ok=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        pooling="cls_mean",
        notes=(
            "Cleanest licence here (MIT, ungated) BUT trained on TCGA. Using it "
            "for a TCGA-derived task contaminates your own evaluation. Avoid for "
            "TCGA cohorts; fine for external/private data."
        ),
    ),
    # ---------------- blocked for commercial use ----------------
    "uni": EncoderSpec(
        name="uni", hf_hub_id="hf-hub:MahmoodLab/UNI", embed_dim=1024,
        licence="CC-BY-NC-ND-4.0", commercial_ok=False,
        notes="Gate agreement bans commercial use of UNI *and models trained on its outputs*.",
    ),
    "uni2-h": EncoderSpec(
        name="uni2-h", hf_hub_id="hf-hub:MahmoodLab/UNI2-h", embed_dim=1536,
        licence="CC-BY-NC-ND-4.0", commercial_ok=False,
    ),
    "prov-gigapath": EncoderSpec(
        name="prov-gigapath", hf_hub_id="hf-hub:prov-gigapath/prov-gigapath", embed_dim=1536,
        licence="Apache-2.0 tag, no-deploy terms", commercial_ok=False,
        notes="Model card: any deployed use case, commercial or otherwise, is out of scope.",
    ),
    "virchow2": EncoderSpec(
        name="virchow2", hf_hub_id="hf-hub:paige-ai/Virchow2", embed_dim=2560,
        licence="CC-BY-NC-ND-4.0", commercial_ok=False, pooling="cls_mean",
    ),
    "h-optimus-1": EncoderSpec(
        name="h-optimus-1", hf_hub_id="hf-hub:bioptimus/H-optimus-1", embed_dim=1536,
        licence="CC-BY-NC-ND-4.0", commercial_ok=False,
    ),
    "phikon-v2": EncoderSpec(
        name="phikon-v2", hf_hub_id="hf-hub:owkin/phikon-v2", embed_dim=1024,
        licence="Owkin non-commercial", commercial_ok=False,
    ),
}

DEFAULT_ENCODER = "h-optimus-0"


class LicenceError(RuntimeError):
    """Raised when a non-commercial encoder is requested without an explicit opt-in."""


def get_spec(name: str) -> EncoderSpec:
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown encoder {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]


def check_licence(name: str, allow_noncommercial: bool = False) -> EncoderSpec:
    spec = get_spec(name)
    if not spec.commercial_ok and not allow_noncommercial:
        raise LicenceError(
            f"Encoder {spec.name!r} is licensed {spec.licence!r} and cannot be used in a "
            f"commercial product. {spec.notes}\n"
            f"Commercially usable alternatives: "
            f"{sorted(k for k, v in REGISTRY.items() if v.commercial_ok)}.\n"
            f"If this is strictly internal research, pass allow_noncommercial=True - but "
            f"note that any weights trained on this encoder's outputs inherit the restriction."
        )
    return spec


class PatchEncoder(nn.Module):
    """Wraps a timm backbone and applies the spec's pooling rule."""

    def __init__(self, spec: EncoderSpec, device: str = "cuda", dtype: torch.dtype = torch.float16):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError("Feature extraction needs `timm`: pip install timm") from e

        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.backbone = timm.create_model(
            spec.hf_hub_id, pretrained=True, num_classes=0, **spec.timm_kwargs
        )
        self.backbone.eval().to(device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, embed_dim]."""
        x = x.to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
            if self.spec.pooling == "cls":
                out = self.backbone(x)
                if out.ndim == 3:                      # some backbones return tokens
                    out = out[:, 0]
            else:
                tokens = self.backbone.forward_features(x)     # [B, 1 + P(+R), C]
                n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
                cls, patches = tokens[:, 0], tokens[:, n_prefix:]
                out = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        return out.float()

    def build_transform(self):
        from torchvision import transforms

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.spec.mean, std=self.spec.std),
        ])


def describe_registry() -> str:
    """Human-readable licence table - printed by the extraction CLI."""
    rows = ["", f"{'encoder':<16}{'dim':>6}  {'commercial':<11}licence", "-" * 72]
    for spec in sorted(REGISTRY.values(), key=lambda s: (not s.commercial_ok, s.name)):
        flag = "YES" if spec.commercial_ok else "NO"
        rows.append(f"{spec.name:<16}{spec.embed_dim:>6}  {flag:<11}{spec.licence}")
    rows.append("")
    return "\n".join(rows)
