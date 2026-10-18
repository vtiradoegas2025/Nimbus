# Building the technical PDFs

The canonical sources are [mathematics.tex](mathematics.tex),
[architecture.tex](architecture.tex) and [studio.tex](studio.tex).
They share the typography in
[guide-preamble.tex](guide-preamble.tex) and vector diagrams under `figures/`.
Edit these LaTeX sources rather than maintaining a second Markdown version.

`solver-coverage.json` maps every solver module and canonical catalog selector
to the guides. When a module or selector changes, update its explanation and
this index, then run `python3 scripts/dev/docs/check_solver_coverage.py --write`
from the repository root. Document builds check coverage and regenerate neither
the catalog nor the scientific implementation.

Install Tectonic, then run from the repository root:

```sh
make docs-pdf
python3 scripts/dev/docs/build_guides.py --check
```

For example, Homebrew provides Tectonic with `brew install tectonic`. Its first
build downloads the required TeX packages/fonts into its cache. Nimbus does not
vendor a TeX distribution. No Markdown conversion step is required.

The PDFs are published as [MATHEMATICS.pdf](../../../../docs/MATHEMATICS.pdf),
[ARCHITECTURE.pdf](../../../../docs/ARCHITECTURE.pdf) and
[STUDIO.pdf](../../../../docs/STUDIO.pdf). Intermediate files and logs go into the
ignored `build/docs/` directory. The build rejects missing glyphs, overflowing
lines and unresolved references, then records source/PDF hashes in
`manifest.json`. Commit the updated source, manifest and PDF together.

Use LaTeX math for units and symbols, and `aligned` environments for long
equations. A visual inspection is still needed after substantial layout changes:
syntactically valid equations can be difficult to read even when they compile.
