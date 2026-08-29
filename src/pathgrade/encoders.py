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

import os
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

# Architectures matching each encoder width, for --random-weights testing.
# Widths are what matter: a mismatch corrupts the output buffer downstream.
RANDOM_ARCH: dict[int, tuple[str, dict]] = {
    1536: ("vit_giant_patch14_dinov2", {"img_size": 224, "init_values": 1e-5}),
    2560: ("vit_huge_patch14_224", {}),        # 1280 CLS + 1280 mean = 2560
    1024: ("vit_large_patch16_224", {}),
    3072: ("vit_giant_patch14_dinov2", {"img_size": 224, "init_values": 1e-5}),
}


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


def resolve_device(device: str = "auto") -> torch.device:
    """Pick an accelerator, including TPU when torch_xla is importable."""
    if device != "auto":
        if device == "xla":
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    except ImportError:
        return torch.device("cpu")


def xla_devices() -> list[torch.device]:
    """Every XLA device this process can address, in order.

    A Kaggle TPU v5e-8 exposes eight, but the extraction loop historically used
    ``xla:0`` alone - one core of eight, which is the single largest piece of
    idle hardware in the job. Returns ``[]`` off TPU so callers can treat the
    single-device case as the default rather than a special case.
    """
    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        return []
    devices = None
    try:
        devices = xm.get_xla_supported_devices()
    except Exception:
        devices = None
    if not devices:
        try:
            devices = [xm.xla_device()]
        except Exception:
            return []
    return [torch.device(str(d)) for d in devices]


def build_encoders(
    spec: EncoderSpec,
    device: str | torch.device = "auto",
    max_devices: int = 1,
    random_weights: bool = False,
    verbose: bool = True,
) -> list["PatchEncoder"]:
    """Build one encoder replica per device, for data-parallel inference.

    Replicas are independent: the encoder is frozen and every batch is
    self-contained, so there is nothing to synchronise and no collective
    communication. That is what makes plain threads sufficient here and keeps
    this far simpler than a distributed training setup.

    Construction is deliberately incremental and forgiving. Each replica costs
    a full set of weights in host RAM before it lands on its device, and a
    partial set is still useful - four cores beat one. If a replica fails to
    build, the ones already built are returned rather than sinking the run.
    """
    primary = resolve_device(device) if isinstance(device, str) else device
    encoders = [PatchEncoder(spec, device=primary, random_weights=random_weights)]
    if max_devices <= 1 or primary.type != "xla":
        return encoders

    # MEASURED 2026-08-22 on a Kaggle TPU v5e-8; the probe script it came from
    # has since been removed, but the run and its numbers are recorded in
    # docs/ENGINEERING.md, "Getting real multi-device throughput did not work":
    # driving several XLA devices from several Python threads in ONE process
    # is not safe. Two threads worked; at four, three of four died, and at
    # eight, seven of eight died, all with
    #
    #   torch_xla/csrc/xla_graph_executor.cpp:691 : Check failed: tensor_data
    #     torch_xla::XLAGraphExecutor::SyncLiveTensorsGraph(...)
    #
    # mark_step() syncs *live tensors on a device*, not just the caller's, so
    # concurrent threads tear each other's in-flight tensors out from under the
    # forward pass. Throughput per surviving thread stayed at 123 patches/s, so
    # this is a correctness failure, not a contention one - and with only one
    # thread surviving, eight "devices" delivered 68 patches/s, slower than one.
    #
    # The supported way to use all eight is one process per device
    # (torch_xla.distributed.xla_multiprocessing.spawn), which is what
    # --shard/--num-shards exist for. Until that lands, degrade loudly rather
    # than fail every slide of a multi-hour run.
    if not os.environ.get("PATHGRADE_FORCE_XLA_THREADS"):
        print(
            f"!! --tpu-cores {max_devices} ignored on XLA: multi-device threading is "
            f"measured-broken (7/8 threads crash in SyncLiveTensorsGraph). Using 1 "
            f"device at ~123 patches/s. Use one process per device for real "
            f"parallelism; set PATHGRADE_FORCE_XLA_THREADS=1 to override.",
            flush=True,
        )
        return encoders

    available = xla_devices()
    # Put the primary first and keep the rest in device order.
    rest = [d for d in available if str(d) != str(primary)][: max_devices - 1]
    for dev in rest:
        try:
            encoders.append(PatchEncoder(spec, device=dev, random_weights=random_weights))
            if verbose:
                print(f"  replica {len(encoders)}/{max_devices} on {dev}", flush=True)
        except Exception as e:  # pragma: no cover - device dependent
            print(f"  !! replica on {dev} failed ({type(e).__name__}: {e}); "
                  f"continuing with {len(encoders)}", flush=True)
            break
    return encoders


class PatchEncoder(nn.Module):
    """Wraps a timm backbone and applies the spec's pooling rule.

    Works on CUDA, TPU (torch_xla) and CPU. On TPU the weights are cast to
    bfloat16 up front rather than relying on autocast: XLA compiles a static
    graph, and keeping the dtype fixed avoids a second compilation.
    """

    def __init__(self, spec: EncoderSpec, device: str | torch.device = "auto", dtype=None,
                 random_weights: bool = False):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover - environment dependent
            raise ImportError("Feature extraction needs `timm`: pip install timm") from e

        self.spec = spec
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.device_type = self.device.type
        self.is_xla = self.device_type == "xla"

        if dtype is None:
            dtype = {"cuda": torch.float16, "xla": torch.bfloat16}.get(self.device_type, torch.float32)
        self.dtype = dtype

        if random_weights:
            # Same shape, no pretrained download. For exercising the pipeline -
            # tiling, decode, device transfer, IO - without a gated weight
            # fetch. Embeddings are meaningless; never train on them.
            #
            # The width has to match the real encoder or the output buffer
            # mismatches. H-optimus-0 is a DINOv2 ViT-g/14 at 1536, which is
            # NOT timm's vit_giant_patch14_224 (1408).
            arch, kwargs = RANDOM_ARCH.get(
                spec.embed_dim, ("vit_giant_patch14_dinov2", {"img_size": 224})
            )
            print(f"!! RANDOM WEIGHTS ({arch}) - pipeline test only, embeddings are garbage")
            self.backbone = timm.create_model(arch, pretrained=False, num_classes=0, **kwargs)
        else:
            self.backbone = timm.create_model(
                spec.hf_hub_id, pretrained=True, num_classes=0, **spec.timm_kwargs
            )
        self.backbone.eval().to(self.device)
        if self.is_xla:
            self.backbone.to(self.dtype)

        # Registered here, outside any inference-mode context. Creating them
        # lazily inside forward() made them *inference tensors* cached on the
        # module; XLA then refused to version-count them and every slide failed
        # with "Cannot set version_counter for inference tensor".
        self.register_buffer(
            "_mean", torch.tensor(spec.mean, device=self.device).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "_std", torch.tensor(spec.std, device=self.device).view(1, 3, 1, 1), persistent=False
        )
        self._verify_width()

    def _verify_width(self) -> None:
        """One probe forward, so a width mismatch fails here and not mid-slide.

        Without this the error surfaces as a numpy broadcast failure deep in
        encode_tiles, once per slide, with nothing pointing at the cause.
        """
        probe = torch.zeros(1, self.spec.patch_px, self.spec.patch_px, 3, dtype=torch.uint8)
        width = int(self(probe).shape[-1])
        if width != self.spec.embed_dim:
            raise RuntimeError(
                f"encoder {self.spec.name!r} produced width {width} but the registry "
                f"declares {self.spec.embed_dim}. Fix the registry or the architecture "
                f"before extracting - a silent width change invalidates saved checkpoints."
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch. Accepts either layout:

        * ``[B, 3, H, W]`` float, already normalised, or
        * ``[B, H, W, 3]`` uint8 straight off the decoder.

        The uint8 path is strongly preferred. Doing ToTensor + Normalize per
        tile on CPU measured ~3.7x more expensive than the actual JPEG decode
        and resize, and it runs inside the decode threads where it competes for
        the very cores that are the bottleneck. Normalising the whole batch on
        the accelerator moves that arithmetic to hardware that is otherwise
        idle waiting for tiles.
        """
        x = x.to(self.device, non_blocking=not self.is_xla)
        if x.dtype == torch.uint8:
            x = self._normalise(x)
        if self.is_xla:
            out = self._forward_inner(x.to(self.dtype))
        else:
            with torch.autocast(device_type=self.device_type, dtype=self.dtype,
                                enabled=self.device_type == "cuda"):
                out = self._forward_inner(x)
        return out.float()

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, W, 3] uint8 -> [B, 3, H, W] normalised float, on device.

        Deliberately free of in-place ops: div_ on a tensor that XLA is tracking
        is the other half of the inference-tensor failure.
        """
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return (x - self._mean) / self._std

    def _forward_inner(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.pooling == "cls":
            out = self.backbone(x)
            if out.ndim == 3:                          # some backbones return tokens
                out = out[:, 0]
            return out
        tokens = self.backbone.forward_features(x)     # [B, 1 + P(+R), C]
        n_prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        cls, patches = tokens[:, 0], tokens[:, n_prefix:]
        return torch.cat([cls, patches.mean(dim=1)], dim=-1)

    def build_transform(self):
        """Per-tile work, kept as cheap as possible: PIL -> uint8 array, nothing more.

        Normalisation deliberately does NOT happen here; see ``forward``.
        """
        import numpy as np

        def to_uint8(tile):
            return np.asarray(tile, dtype=np.uint8)

        return to_uint8


def describe_registry() -> str:
    """Human-readable licence table - printed by the extraction CLI."""
    rows = ["", f"{'encoder':<16}{'dim':>6}  {'commercial':<11}licence", "-" * 72]
    for spec in sorted(REGISTRY.values(), key=lambda s: (not s.commercial_ok, s.name)):
        flag = "YES" if spec.commercial_ok else "NO"
        rows.append(f"{spec.name:<16}{spec.embed_dim:>6}  {flag:<11}{spec.licence}")
    rows.append("")
    return "\n".join(rows)
