"""CrysFDiT-NFT: Crystal Flow Diffusion Transformer with Negative-aware Fine-Tuning.

An all-atom VAE maps every crystal to one latent token per atom plus one global token; a latent
flow-matching diffusion transformer (DiT) generates those tokens. The released DiT was trained on
MP-20, post-trained with Flow-DPO and then with five rounds of DiffusionNFT on verified rewards
(three-MLIP energy above the convex hull and novelty against LeMat-Bulk).

Package layout
--------------
schema    padded batch representation (AtomicSystem / PaddedSystemBatch)
geometry  lattice parameterisation and periodic helpers
latent    latent DiT, flow-matching interpolant, ODE/SDE sampler, latent normaliser
model     all-atom autoencoder and shared transformer blocks
data      atom-count tables and generation contexts
kernel    the frozen chemical-substitution kernel (arm Sall)
pipeline  checkpoint loading and unconditional generation
"""

__version__ = "1.0.0"
