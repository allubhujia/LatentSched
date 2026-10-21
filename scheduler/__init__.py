"""
scheduler — latent-guided schedule search for LatentSched.

Turns the VAE's latent space into a *search space over thread interleavings*:

    latent region  →  decoded thread-index sequence  →  enforced interleaving
                   →  observed bug  →  bandit reward  →  next latent region

Modules
-------
controller    Deterministic, schedule-driven thread scheduler.
targets       Runs the concurrency programs under a fixed schedule.
latent_policy Decodes latent vectors into schedules; bandit over latent regions.
search        CLI entry point for a search campaign.
"""

from .controller import ScheduleController, ScheduleAborted

__all__ = ["ScheduleController", "ScheduleAborted"]
