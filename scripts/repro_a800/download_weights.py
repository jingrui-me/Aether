"""Download only the components used by the official Aether prediction demo."""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ.get("WC_ROOT", "/root/autodl-tmp/worldcache-repro"))
models = [
    ("AetherWorldModel/AetherV1", "6c53ba75e398c8b91623ed86757b2289eb45f1ce", ["transformer/*"]),
    ("THUDM/CogVideoX-5b-I2V", "a6f0f4858a8395e7429d82493864ce92bf73af11", ["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*"]),
]
for repo, revision, patterns in models:
    snapshot_download(
        repo_id=repo,
        revision=revision,
        allow_patterns=patterns,
        local_dir=root / "models" / repo.split("/")[-1],
        max_workers=3,
    )
