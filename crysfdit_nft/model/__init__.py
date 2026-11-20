"""All-atom autoencoder (``autoencoder``) and the transformer blocks it shares with the latent DiT (``student``).

Nothing is imported here: ``latent`` imports ``model.student`` and ``model.autoencoder`` imports ``latent``, so an
eager import would create a cycle.
"""
