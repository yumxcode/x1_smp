"""Gait analysis of an allmimic motion clip, from foot-sole heights alone.

One implementation, imported by both `scripts/build_x1_motions.py` (to write the per-clip metrics) and
`tools/x1_motion_audit.py` (to re-derive them). Two independent implementations of "where is the step" disagreed
badly once (16/41 vs 41/41 clips resolving both feet), so there is exactly one now.

Why hysteresis and not a single threshold. The clips are grounded per frame, so the stance sole sits at z = 0 by
construction and the swing foot rises only 3-9 cm; a sole hovers near any single threshold for several frames and a
bare trigger counts one step as several. Each foot therefore uses two levels: it becomes airborne above `AIR`, and
returns to contact only below `GROUND` (a Schmitt trigger), with a minimum contact run to reject flicker.

Cadence is reported as the median inter-touchdown interval (robust when one touchdown is missed) plus the count, so
a clip either has a usable period or says so explicitly. Alternation counts opposite-foot touchdowns inside each
same-foot cycle.
"""

from __future__ import annotations

import numpy as np

AIR = 0.02  # m above the lower sole: definitely airborne
GROUND = 0.01  # m above the lower sole: definitely in contact
MIN_CONTACT_FRAMES = 3
# A swing phase shorter than this is a weight shift / a toe tap, not a step. At 50 Hz, 5 frames = 0.1 s; the clips
# whose swing clearance is only 3-4 cm (measured: 35_01 at 3.2 cm) lift for 2-4 frames, and counting those turned one
# stride into a burst (2.38 Hz for a clip that visibly walks at ~0.9 Hz).
MIN_PHASE_FRAMES = 5
# a stride inside this band, with both feet represented, is what "walk-like" means for this library
STRIDE_BAND_S = (0.5, 1.5)


def touchdown_events(sole: np.ndarray, air: float = AIR, ground: float = GROUND,
                     min_contact: int = MIN_CONTACT_FRAMES, min_phase: int = MIN_PHASE_FRAMES):
    """(N,2) sole heights -> list of touchdown frame indices per foot, using hysteresis.

    A foot is airborne once its height exceeds `air`; it touches down when it drops below `ground` again. Runs
    shorter than min_phase/min_contact frames are treated as noise and ignored, which keeps a hovering sole from
    producing a burst of events.
    """
    lower = sole.min(axis=1, keepdims=True)
    rel = sole - lower  # 0 for the lower foot, positive for the swing foot
    events = []
    for k in range(sole.shape[1]):
        h = rel[:, k]
        state_air, out, i, n = False, [], 0, len(h)
        while i < n:
            if not state_air:
                if h[i] > air:
                    j = i
                    while j < n and h[j] > ground:
                        j += 1
                    if j - i >= min_phase and j < n:
                        m = j
                        while m < n and h[m] <= ground:
                            m += 1
                        if m - j >= min_contact:
                            out.append(j)
                    state_air = True
                    i = j
                else:
                    i += 1
            else:
                if h[i] <= ground:
                    state_air = False
                i += 1
        events.append(np.array(out, dtype=int))
    return events


def gait_summary(sole: np.ndarray, fps: float) -> dict:
    """per-clip gait dict: touchdown counts, cadence per foot, alternation, swing clearance, contact fraction"""
    ev = touchdown_events(sole)
    rel = sole - sole.min(axis=1, keepdims=True)
    cadence, strides = [], []
    for k in (0, 1):
        d = np.diff(ev[k])
        # one stride is enough to state a period; a clip is 2-4 s long, so requiring two would throw away most of them
        strides.append(None if len(d) == 0 else round(float(np.median(d) / fps), 3))
        cadence.append(round(float(fps / np.median(d)), 2) if len(d) else None)
    alt = [int(((ev[1] > a) & (ev[1] < b)).sum()) for a, b in zip(ev[0][:-1], ev[0][1:], strict=False)]
    count = [int(len(ev[k])) for k in (0, 1)]
    both = count[0] >= 2 and count[1] >= 2
    swings = float(rel.max())
    walk_like = bool(
        both and swings >= AIR
        and all(c is not None and 1 / STRIDE_BAND_S[1] <= c <= 1 / STRIDE_BAND_S[0] for c in cadence)
        and (not alt or sum(1 for a in alt if a == 1) >= max(1, int(0.6 * len(alt))))
    )
    return {
        'touchdowns': count,
        'stride_s': strides,
        'cadence_hz': cadence,
        'alternation_between_left_td': alt,
        'swing_clearance_max_m': round(swings, 4),
        'other_foot_contact_frac': [round(float((rel[:, k] <= GROUND).mean()), 3) for k in (0, 1)],
        'gait_resolved': bool(both),
        # the detector needs the swing to clear the hysteresis band; below that the cadence it reports is a weight
        # shift, not a step, and the clip is counted as "not interpretable" rather than silently averaged in
        'walk_like': walk_like,
    }


def report_line(g: dict) -> str:
    c = g['cadence_hz']
    both = g['gait_resolved']
    return (f"td {g['touchdowns']} cadence {c} stride {g['stride_s']}s "
            f"alt {g['alternation_between_left_td']} swing {100 * g['swing_clearance_max_m']:.1f}cm "
            f"{'both feet' if both else 'ONE FOOT ONLY'}")
