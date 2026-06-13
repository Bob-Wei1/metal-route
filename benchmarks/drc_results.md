# DRC results

Tracks the native design-rule-check (`mr-drc`) violation counts on routed boards
as the DRC-clean routing milestone progresses. The checker is scoped to the
violation classes metalroute can produce — copper↔copper **clearance**,
**via-through-plane** (a through-via barrel crossing a foreign poured plane), and
via **annular-ring** — not a full KiCad-equivalent DRC. A best-effort `kicad-cli pcb
drc` cross-check validates these classes against KiCad ground truth.

How to reproduce:

```
metalroute drc --input bench/fixture_fresh/fixture.dsn \
  --skip-nets=GND --skip-nets=+5VA --skip-nets=-5VA --skip-nets=3V3 \
  --max-violations 30 --out benchmarks/drc_baseline.json
```

`--skip-nets` drops the four poured power nets (they are planes, not routable
traces); the planes still exist physically, so signal vias are correctly seen
drilling through them.

## M1 baseline — `bench/fixture_fresh/fixture.dsn`

8-layer board (4 signal: F.Cu/In2.Cu/In5.Cu/B.Cu, 4 power planes: In1.Cu=GND,
In3.Cu=3V3, In4.Cu=-5VA, In6.Cu=+5VA), 0.45/0.2 mm through-vias, clearance rule
0.15 mm. Routed with the negotiated CPU router.

| Metric | Value |
| --- | --- |
| Two-point nets routed | 142 / 142 |
| Original nets fully connected | 55 |
| Vias | 244 |
| **DRC violations (total)** | **3019** |
| — clearance | 2715 |
| — via-through-plane | 304 |
| — annular-ring | 0 |

The two root causes are now measurable and attributable:

- **via-through-plane = 304.** Every signal via is a physical F.Cu↔B.Cu through-via
  with no antipad, so its barrel shorts each inner GND/3V3/±5VA plane it drills
  through (unless it happens to be that plane's own net). This is the dominant
  *plane-short* class.
- **clearance = 2715.** Routed copper reserves no clearance halo and via pads are
  0.45 mm, so tracks and via pads bleed into adjacent grid cells owned by other
  nets — signal-to-signal shorts and tight-clearance breaches. (annular-ring is 0:
  the 0.45/0.2 via gives a 0.125 mm ring, above the 0.05 mm minimum.)

These are exactly the targets for the next milestone (clearance-aware net-ownership
halos, via keepout, and full-stackup plane antipads). Re-run the command above after
each fix and append a row here as the totals fall toward zero.

## M2 — DRC-clean routing (first pass)

Three mechanisms landed: net-ownership **clearance halos** + **via keepout** in the
negotiated router's legalization fold (`mr-cpu`), and **plane-antipad modelling** for
poured-zone boards (`mr-cli/drc.rs`). Same command + fixture as above
(`benchmarks/drc_after_m2.json`).

| Metric | M1 baseline | After M2 | Δ |
| --- | --- | --- | --- |
| Two-point nets routed | 142 / 142 | 108 / 142 | −34 |
| Fully-connected nets | 55 | 39 | −16 |
| **DRC violations (total)** | **3019** | **1394** | **−54 %** |
| — via-through-plane | 304 | **0** | **eliminated** |
| — clearance | 2715 | 1394 | −49 % |
| — annular-ring | 0 | 0 | — |

What worked and what's left:

- **via-through-plane 304 → 0.** Poured GND/3V3/±5VA zones relieve a foreign
  through-via with an antipad; modelling that (`antipad_radius = drill/2 +
  plane_antipad`) clears the class. This assumes the planes are *poured zones* — the
  `kicad-cli pcb drc` cross-check (M1/M2 follow-up) is what confirms the fabricated
  board really reliefs them. Run `drc --no-plane-zones` for the pessimistic
  bare-copper model (which still reports all 304).
- **clearance 2715 → 1394 (−49 %).** Net-owned halos force different nets ≥ clearance
  apart in legalization. Two limits keep it from reaching 0: (1) the halo is *not*
  stamped over foreign pads (so a net can still reach its own pad), so track-to-pad
  clearance isn't fully reserved; (2) at the bounds-derived grid resolution a
  Chebyshev cell halo under-covers diagonal and 0.45 mm via-pad cases that the
  continuous checker measures exactly.
- **Connectivity cost: 142 → 108 routed.** The router now *refuses to short* — where
  it can't find a clearance-respecting path it drops the net rather than overlap. That
  is the correct trade, but recovering the lost nets needs a finer grid and clearance
  pricing in the negotiation phase (not just legalization).
- **Runtime** rose sharply (≈5 s → minutes): nearly every negotiated path is now
  "dirty" and reroutes around halos. This is the strongest motivation for the M3
  Metal acceleration (cross-net batching + GPU clearance stamping).

Next: push clearance toward 0 (pad-aware halos + finer resolution + negotiation-phase
clearance) and recover connectivity, then GPU-accelerate the now-dominant routing and
DRC costs.
