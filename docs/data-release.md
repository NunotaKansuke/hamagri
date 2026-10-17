# Data release

## Software and atlas are separate records

The `hamagri` source repository contains software, examples, and documentation.
The full map atlas is a separate data product because its stored arrays are
about 72 GB. The intended data record is one curated atlas with one DOI; the
software repository must not contain copies of the coefficient arrays.

## Canonical atlas

The intended source is the curated, integrated version-1 atlas, with root
`manifest.json` and a `buckets/` directory. The latest inspection recorded
33,993 maps, 161 buckets, and `m_max=512`. Root manifests use
`hamagri-bucketed-fourier-maps`; child manifests use `hamagri-fourier-maps`.
The public code and distributed atlas use the same `hamagri` manifest schema.

The atlas stores Fourier coefficients, reconstruction-error metadata, and
radial-layout metadata. It does not need chi-square certificate files. The
public loader opens the version-1 bucket layout and does not require
`certified_error*.npy` files. The public search returns a profiled `chi2` score
and risk diagnostics, without certified chi-square bounds.

## Verified package and Zenodo constraints

The staging package was prepared from the designated integrated version-1
atlas on 2026-10-17. The original file inventory contains 206,093 paths
totaling 71,562,006,294 bytes. The public staging copy uses the `hamagri-*`
manifest identifiers expected by the software. Its updated `SHA256SUMS.tsv`
has SHA-256
`68268c29b1fccbcecb36850572369dd249e507155efc5ecaf79946a85e5de3f8`.
Four uncompressed tar archives preserve the root `manifest.json` and the
relative `buckets/` layout. All 206,093 archived files were streamed through a
size and SHA-256 comparison against the inventory; every archive checksum also
matched. No paths were missing, duplicated, unexpected, or non-regular.

For the public staging copy, 162 JSON manifests were updated in place with
equal-length identifier replacements. The coefficient arrays were not
rewritten. The inventory rows for those manifests and all four archive
checksums were regenerated; the other 205,931 archived files are unchanged.

The archive sizes are 21,111,152,640; 21,584,322,560; 21,623,050,240; and
7,678,863,360 bytes (71,997,388,800 bytes total). The upload package has four
data files, plus the README, data-license notice, both checksum files, and
`verify_inventory.py`.
Temporary bucket lists and internal build metadata are not part of the upload
package. All four tar files were extracted into a fresh directory with
archived mtimes normalized for portability, then checked using the packaged
`verify_inventory.py`. The restored tree contained 206,093 files and
71,562,006,294 bytes, with zero verification errors. The temporary 72 GB
restored copy was removed after the check.

Zenodo currently documents a default per-record total of 50,000,000,000 bytes,
up to 100 uploaded files, and up to 50GB per file. Its quota page documents up
to 150GB of additional storage that can be assigned across records, up to
200GB total for a record. The nine-file package totals 72,023,939,465 bytes,
so it needs at least 22,023,939,465 bytes of additional quota. The actual
available quota is account-specific and must be checked in the account's
storage settings before upload. See [Zenodo file limits](https://help.zenodo.org/docs/deposit/manage-files/)
and [Zenodo storage quota](https://help.zenodo.org/docs/deposit/manage-quota/).

## Remaining before publication

- Use one creator on the record; confirm the public display name and ORCID, if
  any, and confirm the software copyright line. The selected licenses are MIT
  for the software and CC BY 4.0 for the atlas. Confirm the creator has
  authority to license the atlas contents before publication.
- Confirm account quota, citation text, and the exact `hamagri` software
  release used to read the atlas.
- Create a Zenodo draft, upload the nine package files, download and verify
  the uploaded archives, and review the record before publication.

No Zenodo draft, upload, DOI, or publication has been created.
