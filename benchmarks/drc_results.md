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
