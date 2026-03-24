This directory contains vendored third-party dependencies used by UNAGI.

Current contents:
- `idrem/`: upstream iDREM runtime and assets from the iDREM project

Notes:
- These files are not authored by the UNAGI project.
- Keep the upstream license and attribution files alongside the vendored copy.
- UNAGI prefers this vendored runtime automatically, but users can still override it with
  `run.idrem_dir` or `UNAGI_IDREM_DIR`.
