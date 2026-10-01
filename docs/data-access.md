# Dataset rights and record redistribution

Status checked on 2026-10-01. The source datasets are not included in this
package. The derived per-flow records are needed to rebuild the manuscript's
reported table and figures; they are not released under the software license.
The current public-release proposal is **not cleared for redistribution** until
the permissions below are resolved in writing or the affected records are
removed and the manuscript's reproducibility claim is revised.

## ToN-IoT

The official [UNSW ToN-IoT dataset page](https://research.unsw.edu.au/projects/toniot-datasets)
grants free use for academic research and says commercial use is allowed after
asking the dataset author. It asks users to cite the listed dataset papers. It
does not expressly state whether a public repository may redistribute the
row-level per-flow measurements derived from ToN-IoT, or whether those records
may be accessed and reused commercially. Public posting would make the files
available beyond academic users, so this package does not infer permission from
the research-use grant alone.

Required clearance: written permission for public redistribution of the
ToN-IoT-derived per-flow result records, including access by commercial users,
with any citation, notice, and downstream-use conditions specified by the
rights holder.

## Corrected CIC-IDS2017

The experiment configuration identifies the input as the corrected
`CICIDS2017_improved` release distributed with the CNS 2022 study by Liu,
Engelen, Lynar, Essam, and Joosen. The [CIC-IDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html)
and [CIC FAQ](https://www.unb.ca/cic/datasets/index.html) state that the
original CIC dataset may be redistributed, republished, and mirrored with the
required citations. The configured input is the corrected release hosted by
the CNS study's [supplementary site](https://intrusion-detection.distrinet-research.be/CNS2022/),
not just the original UNB download. I found no explicit redistribution license
for that corrected package or for derived per-flow records on the supplementary
pages. The original CIC permission therefore is not treated as clearance for
the corrected variant or these outputs.

Required clearance: written permission or an explicit license from the
corrected-release maintainers covering public redistribution of the derived
per-flow result records and any attribution conditions.

## Scope of licenses in this repository

`LICENSE` grants MIT rights only for original project software in `src/`,
`scripts/`, `tests/`, and `configs/`. It does not grant rights to source data,
per-flow records, aggregate outputs, figures, the manuscript, or third-party
material. Those materials have no additional license grant here; their
copyright and data-provider terms remain in force. `CITATION.cff`'s MIT value
refers to the original software only. The manuscript, figures, and records
must be cleared separately with their relevant authors, contributors, or
publisher before the package is made public.

The package includes both dataset tasks' per-flow records only to support the
current reconstruction claim. If either clearance is denied or remains
unanswered, those records must stay out of the public snapshot and the paper
must no longer claim complete record-based reconstruction from the repository.
No raw captures or source feature matrices are included.
