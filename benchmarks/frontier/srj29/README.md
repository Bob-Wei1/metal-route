# SRJ29 AM62L / LPDDR4 frontier fixture

This target is deliberately kept outside `benchmarks/corpus`: that historical
112-board baseline remains immutable, while frontier boards can evolve under
separately reported rules and performance expectations.

`sample021-am62l-lpddr4.srj.json` is a routing-only normalization of
`samples/sample021.json` from
[`ShiboSoftwareDev/dataset-srj29-ddr3-bga-pairs`](https://github.com/ShiboSoftwareDev/dataset-srj29-ddr3-bga-pairs)
at commit
[`585c3abd4c6ce4be52bbc430b1d5e6dcc9b4ea30`](https://github.com/ShiboSoftwareDev/dataset-srj29-ddr3-bga-pairs/commit/585c3abd4c6ce4be52bbc430b1d5e6dcc9b4ea30).

The normalization retains the exact eight-layer board geometry, 573 pads, 33
connections, bus and differential-pair declarations, layer data, and physical
design-rule fields. It drops presentation and per-pad source metadata that do
not affect the SimpleRouteJson routing problem. The source dataset is MIT
licensed; its license is reproduced in `LICENSE`.

Source SHA-256: `38c774814decccc3182fdad3735e77489a0cd7a6b3770793a55ae37608ccfd67`.
Normalized fixture SHA-256:
`d7f53f9f2aff69379530b7ba6703a7bb583337e381a215f409309cf6689a19eb`.
