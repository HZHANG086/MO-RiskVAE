# Release inspection — 28 October 2026

## Scope and method

Inspected the supplied local `MO-RiskVAE.zip` without executing its code or loading checkpoints. The outer archive contains 1,303 non-directory entries totaling 2,118,372,932 uncompressed bytes, including macOS sidecars. Four non-sidecar nested ZIPs were enumerated and their small text members scanned. Selected source files were inspected for imports, entry points, command-line options, local paths, serialization, credentials and patient identifiers. Source-only publication uses an explicit allowlist; excluded binary/data artifacts were not treated as safe based on their names.

## Findings and disposition

- Clinical labels contain `PUBLIC_ID` and survival outcome fields; processed feature and latent-vector tables also use sample identifiers. All CSVs, spreadsheets, external cohort matrices and per-patient outputs are excluded. Public availability and de-identification were not independently established.
- Checkpoints, figures, logs, PDFs, DOCX files, review material and nested output archives are excluded. No binary patient artifact is published.
- A file named `batchCorrectedData.R` is actually gzip-compressed serialized R workspace data (`RDX2`), not source. It is excluded without deserializing it.
- Old `.git` repositories, their configurations/history, IDE state, caches, macOS resource forks and unrelated ChatMOF/MOFFUSION content are excluded.
- Credential pattern scans found assignment-like examples only in excluded ChatMOF documentation/metadata. No recognized GitHub/OpenAI/AWS credential pattern or private-key header was found in the selected source. This is a bounded static scan, not a guarantee that every possible secret format is detected.
- The selected source contains no matches for checked MMRF/GSM sample identifier patterns. Patient field names and file-reading code remain because they are part of the research implementation; no data tables are included.
- Retained professional attribution includes the upstream maintainer contact in original package metadata. No user-specific private contact or authentication material is added.
- No individual released file approaches GitHub's large-file limit; this is a small, text-only source release.

## Code integrity and validation

`SOURCE_MANIFEST.json` records the source path, SHA-256 and byte count for each of the 87 preserved source/configuration/license files. These files are byte-for-byte copies, including original comments and inactive experimental implementations. New documentation, `.gitignore` and short variant README pointers are separate additions. No model behavior was changed.

All 73 Python source files parse successfully with Python's AST parser. All six shell launchers pass `bash -n`. All release files decode as UTF-8. Remaining content was scanned again for credential signatures and checked sample identifier patterns.

Runtime dependency installation, training, inference and numerical reproduction were not performed. Missing baseline/MMD source, incomplete HSIC entry points, incompatible launcher options, original server paths and the missing annotation reference are documented in the README. Existing `eval` parameter reconstruction and checkpoint deserialization are retained; use trusted inputs only. Event/censoring conventions differ across retained scripts and must be reconciled before using their scientific results.

The paper title, authors and identifier were verified against https://arxiv.org/abs/2604.06267. Its topic aligns with the supplied experiment families; an exact correspondence between every source snapshot and every reported result has not been established.
