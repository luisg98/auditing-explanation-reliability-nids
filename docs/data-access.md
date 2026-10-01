# Dataset terms and package scope

Review checked on 2026-10-01. This package does **not** contain the ToN-IoT or
CIC-IDS2017 source datasets, packet captures, source feature rows, or the
third-party `CICIDS2017_improved` archive. It contains project-generated
analysis records and aggregate results used to reproduce the paper's table
and figures. Direct source locators and fingerprints have been removed; the
per-flow identifiers are random and have no retained crosswalk.

## ToN-IoT

The [official UNSW ToN-IoT page](https://research.unsw.edu.au/projects/toniot-datasets)
grants free use for academic research, asks users to cite the listed dataset
papers, and says commercial use of the dataset is allowed after asking its
author. This project uses ToN-IoT for academic research and shares derived
research results; it does not mirror or redistribute ToN-IoT source records.
The commercial-use condition applies to use of the source dataset, and this
package does not grant rights to that dataset. Anyone obtaining ToN-IoT for a
new experiment must follow the current UNSW terms and cite the papers listed
on the upstream page.

## CIC-IDS2017

The [UNB CIC FAQ](https://www.unb.ca/cic/datasets/index.html) says its datasets
may be redistributed, republished, and mirrored in any form, with citations
to the dataset and its paper. The experiment used the corrected
`CICIDS2017_improved` distribution from the [CNS 2022 study site](https://intrusion-detection.distrinet-research.be/CNS2022/),
but this package does not redistribute that archive or the underlying
CIC-IDS2017 source files. The corrected archive's terms would need separate
review only if that archive or its source rows were added to this package.

## License scope

`LICENSE` grants MIT rights only for original project software in `src/`,
`scripts/`, `tests/`, and `configs/`. It does not license benchmark datasets,
the derived result records, the manuscript, figures, or third-party material.
The package contains no benchmark source data. A separate license for
project-generated records or paper materials should be added only after the
authors confirm they hold the rights to grant it; the GitHub license badge
does not extend the MIT grant beyond the listed software directories.

## Review conclusion

For the current package scope, the dataset terms do not create a separate
permission blocker: the upstream datasets are used for academic research, the
package contains derived results rather than source data, and the required
dataset attribution must remain documented. This conclusion does not cover
adding raw or corrected benchmark files, source feature rows, or any data from
another source. The separate review of record linkability and of author or
publisher rights for the manuscript and figures remains necessary.
