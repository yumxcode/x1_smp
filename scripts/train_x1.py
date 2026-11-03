"""Train wrapper: registers SMP-X1 tasks then delegates to mjlab.scripts.train.main.

Same pattern as the SUZ upstream scripts/train.py -- the side-effect import of
smp.rl.tasks registers Smp-Forward-X1 in the mjlab task registry.
"""

from __future__ import annotations

from mjlab.scripts.train import main

import smp.rl.tasks  # noqa: F401  # registers Smp-Forward-G1 / Smp-Forward-X1

if __name__ == "__main__":
    main()
