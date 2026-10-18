# Literature Notes

## Klemme & Amrouch - Deep-learning cell-delay modeling for STA

- Title used in this project: *Deep-learning cell-delay modeling for static timing analysis*
- Venue: *Ain Shams Engineering Journal* (Elsevier)
- Publication year recorded by the project: 2022
- Publisher page: <https://www.sciencedirect.com/science/article/pii/S2090447922001393>
- Verification status: bibliographic details and current citation metrics still require
  confirmation against the publisher/Crossref/Scopus. Live lookup was unavailable on
  2026-09-02, so no impact factor, quartile, or citation count is recorded here.

### Venue assessment (2026-09-02)

This is a formal peer-reviewed journal article and therefore has stronger publication
status than an arXiv-only preprint. The journal is a broad engineering venue rather than
a core EDA venue. For recognition specifically within static timing analysis and cell
library characterization, specialist venues such as IEEE TCAD, ACM/IEEE DAC, ICCAD, DATE,
and ASP-DAC normally carry greater field-specific weight. "IEEE" is a publisher/society,
not a single quality tier; an IEEE article must be judged by its particular journal or
conference. Recommended use: retain this paper as highly relevant direct prior work, but
triangulate important claims with specialist EDA publications.

### Relationship to this repository

Verdict: **directionally guided by the paper, but not a reproduction of it**.

Confirmed alignment from the repository:

- Both treat cell delay as a learnable function and use an ANN/MLP as an alternative to
  conventional NLDM table evaluation.
- The local `MLPNldm` baseline accepts input slew and output load, the two independent
  variables of an NLDM table.
- The project evaluates ANN predictions against a conventional interpolation baseline.
- OpenSTA integration is an explicit end-to-end validation goal.

Important differences and extensions in the current main research design:

- The legacy per-table sparse-grid MLP pilot is closest to a direct DL-NLDM-style study,
  but its audited result was negative: the 64-64 per-table MLP was worse than regular-grid
  bilinear interpolation at every tested sparsity. It must not be reported as ANN
  superiority.
- The thesis-level model is feature-conditioned across cells and PVT conditions. It uses
  inference-available structural, Boolean, drive-strength, pin-capacitance, Vt, process,
  voltage, and temperature features rather than relying on cell/corner ID embeddings.
- The main evaluation includes random-point, unseen-cell, unseen-corner, and joint-unseen
  splits, with group-disjoint leakage checks.
- Five seeds, micro and per-cell macro aggregation, MAE/RMSE/sMAPE/tail relative error/R2,
  confidence intervals, immutable manifests, hashes, and provenance are required.
- External-technology validation is planned, making the claim broader than interpolation
  within one characterized library.

### Completion boundary

As of 2026-09-02, the repository-side experiment harness and split contracts are prepared,
but formal GPU training has not started. OpenSTA end-to-end path-delay/slack validation and
external-technology validation are also pending. Consequently, the project may state that
the cited paper motivated the approach and informed the baseline, but it must not yet claim
that the paper's method was fully reproduced or that the proposed extension is validated.

### Evidence limitation

This comparison is based on the repository's recorded citation and implemented research
artifacts. The paper's full text was not locally available and its exact architecture,
training split, loss, dataset, and STA validation protocol have not yet been audited
section-by-section. A future paper matrix should compare those fields before making a
method-level equivalence or novelty claim.
