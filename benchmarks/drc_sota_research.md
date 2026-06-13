# DRC-clean routing — SOTA research + roadmap

Synthesis of two research passes (TritonRoute/OpenROAD + academic detailed routing;
Freerouting + shape-based + Specctra clearance semantics) into an actionable plan for
our grid-based, PathFinder-style negotiated-congestion router. Citations at the end.

## The core lesson

Both production anchors agree: **design-rule clearance must live in the path-search
cost, not in a post-hoc legalization step.** A hard clearance block (what M2 does)
fails two ways — it *drops* nets that can't honor clearance, and it produces a
violation count that can't be iterated down. The two reference designs:

- **TritonRoute (gridded, closest to us):** clearance is a *soft additive cost* on the
  A* edges. Committing a shape stamps an expanding halo of `objCost` (~8× base edge
  cost) computed by speculatively placing a "shadow" wire/via and checking spacing;
  a second `markerCost` (~64×, with decayed history that's never fully removed) is
  added where real DRCs appeared — negotiated congestion applied to *violations*.
  Object cost supports **same-net overriding** (a net pays no clearance against its own
  geometry). Pins get ≥3 **access points**; before routing net *i* every unrouted
  foreign pad's access is *priced as occupied*, then the current net's own access is
  un-reserved (`reservePA`/`unreservePinAccess`) — clearance from foreign pads without
  blocking their owner. Result: <20 DRCs on 7/10 ISPD-2018 cases (~99% reduction).
- **Freerouting (gridless, shape-based):** inflates every foreign item by its clearance
  (Minkowski sum) and searches only the residual free space, so paths are
  clearance-legal *by construction*; same-net items are *targets*, not obstacles.
  Clearance is an N×N **ClearanceMatrix** over item classes (`pin/smd/via/wire/area`),
  per layer. Its only violation source is the post-route pull-tight optimizer moving
  copper without re-checking — the exact failure mode of a stale legalization halo.

## Prioritized roadmap for our router

Our three M2 failure modes map directly: (a) halo can't reserve over foreign pads
without blocking their owner → **P2 pin-access reservation**; (b) coarse Chebyshev
halo under-covers diagonal/0.45 mm-via geometry → **half-pitch grid**; (c) ~24% nets
dropped because clearance isn't priced during the search → **P1 soft clearance cost**.

- **P1 (highest impact). Soft clearance, not a hard drop.** Make the clearance halo a
  high *cost* in the search, never a hard block — only foreign *copper* (path cells)
  stays a hard block. Nets then route through a halo at penalty (a recorded, then
  iteratively-minimized violation) instead of being dropped. Add a `markerCost`-style
  history term on cells that ended up in a real DRC, decayed but not removed, so a
  rip-up-and-reroute loop drives violations toward zero. Fits our existing
  present/history pricing with minimal structural change.
- **P2. Per-net ownership + pin-access reservation.** Tag halo cells with their owner
  net/group (we already own path cells); own-net halo costs zero, foreign halo costs
  the P1 penalty. Price not-yet-routed foreign pads' access as occupied; un-reserve the
  current net's own pad before its search.
- **P3. Per-type clearance** from the DSN `(rule (clearance N (type wire_wire|via_via|
  smd_smd|...)))` — at least split wire/wire, wire/pad, via/via; the residual
  track-to-pad violations suggest pad clearance ≠ track clearance in the source.
- **P4. Half-pitch grid** (or off-grid lines only through pad access points) to cover
  diagonal/sub-cell geometry the cell halo rounds away. Costs ~4× cells; do after P1/P2.
- **P5. DRC-repair rip-up pass.** A true geometric (continuous, via mr-drc) check after
  routing; rip up the cheaper net of each residual violation and reroute under correct
  constraints. The safety net that reaches zero after P1–P3 reduce by an order of
  magnitude.

Recommended sequence: **P1 → P2 → P3 → P4 → P5.** P1+P2 target the dropped nets and the
foreign-pad problem with the least structural change and have the strongest evidence
(DRAPS 96% / TritonRoute ~99% DRC reduction come specifically from soft,
design-rule-aware, ownership-respecting search cost).

## Sources

- TritonRoute (Kahng/Wang/Xu): https://vlsicad.ucsd.edu/Publications/Journals/j133.pdf
- OpenROAD `drt`: https://openroad.readthedocs.io/en/latest/main/src/drt/README.html
- DRAPS (design-rule-aware path search, 96% DRC reduction): https://ieeexplore.ieee.org/document/8815877/
- PathFinder cost `c(n)=(b(n)+h(n))·p(n)`: https://dl.acm.org/doi/pdf/10.1145/201310.201328
- "Tao of PAO" pin access, DAC 2020 (TritonRoute ref).
- Freerouting architecture (ClearanceMatrix, AutorouteEngine, push-and-shove): https://deepwiki.com/freerouting/freerouting
- Freerouting routing options (clearance classes, neckdown, pad-to-turn-gap): https://freerouting.org/freerouting/manual/routing-options
- SPECCTRA DSN reference (rule/clearance/class grammar): https://cdn.hackaday.io/files/1666717130852064/specctra.pdf
- KiCad interactive router (push-and-shove, same-net finishes, neckdown): https://github.com/KiCad/kicad-doc/blob/master/src/pcbnew/pcbnew_interactive_router.adoc
- ISPD-2018 detailed-routing contest: http://www.ispd.cc/contests/18/index.htm
