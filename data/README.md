# NOAA data

`raw/` contains the complete Hartford Bradley (USW00014740) station files downloaded from NOAA on November 4, 2026. The files retain all 329 original columns. Gzip compression changes only their storage format; decompression restores the original bytes.

| File | Original records | Compressed bytes |
|---|---:|---:|
| [2023](raw/GHCNh_USW00014740_2023.psv.gz) | 12,855 | 1,006,152 |
| [2024](raw/GHCNh_USW00014740_2024.psv.gz) | 13,677 | 1,054,818 |
| [2025](raw/GHCNh_USW00014740_2025.psv.gz) | 13,520 | 1,015,722 |
| Total | 40,052 | 3,076,692 |

[manifest.json](raw/manifest.json) records the source URLs, download date, time coverage, row counts, and SHA-256 checksums for both the original and compressed files. `pilot.py` verifies the original-byte checksums before using the data. No separate data download is needed after cloning this repository.

The root-level `raw_samples_20.psv` is a small preview of the first 20 records of the 2023 file. It is not the complete training data.

## Quality policy

The pilot retains numeric temperatures with valid timestamps for the selected station and these source-specific quality codes:

- Sources 313, 314, 315, 322, 335, 343, 344, and 346: `0`, `1`, `4`, `5`, or `9` (documented passing checks).
- Sources 220, 221, 222, 223, 347, and 348: `1` (good).

Other codes and unrecognized sources are excluded, including blank quality codes. This deliberately conservative policy can be reviewed later against NOAA's documentation. Duplicate station/timestamp records retain the last valid row in the source order. The report logs each removal stage.

Hourly bins are `(h − 1 hour, h]`, labeled by their ending hour `h`. The last valid report in a bin supplies its temperature. Empty bins stay missing. Bins that reach across a partition boundary are excluded. The first usable hourly endpoint in each partition is one hour after its start; all input endpoints and targets must be before its exclusive end.

## Source and license

Menne et al. (2023), *Global Historical Climatology Network-Hourly (GHCNh)*, NOAA NCEI, [doi:10.25921/jp3d-3v19](https://doi.org/10.25921/jp3d-3v19). Subset: USW00014740, 2023–2025; accessed November 4, 2026.

- [NOAA documentation](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf)
- [NOAA metadata and CC0-1.0 data license](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01688)
