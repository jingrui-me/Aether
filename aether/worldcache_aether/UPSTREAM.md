# WorldCache source attribution

Source: https://github.com/FofGofx/WorldCache
Commit: b921368f7dfbd7ca5d7cfcd0276fec1d1cfd7d91
License: Apache-2.0; see LICENSE in this directory.

This package and the WorldCache modifications in aether/pipelines/aetherv1_pipeline_cogvideox.py come from that snapshot.

Local compatibility change: the inference call to CogVideoXBlock omits attention_kwargs because Diffusers 0.32.2 does not accept that keyword. This reproduction uses the official demo with attention_kwargs=None. The WorldCache prediction and skip-decision algorithms are unchanged.
