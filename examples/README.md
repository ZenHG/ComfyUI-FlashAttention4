# Examples

No workflow JSON is shipped: a runnable graph needs model files (UNET / CLIP / VAE)
that are not part of this repo.

To try the node:

1. Add `FlashAttention4 Model Patch` to any MODEL chain.
2. Put it after the loader / LoRA / sparse-attention pack, before the sampler.
3. Add `FlashAttention4 Stats` with `reset=True` and read the outputs after the run.

A valid run has `fa4 > 0` and all `sdpa_*` / `kernel_errors` = 0.
