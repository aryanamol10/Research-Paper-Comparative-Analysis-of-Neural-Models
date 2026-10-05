# Paper: Depth, Breadth, or Modularity?

This folder is the IEEE conference version of the hybrid architecture study, laid out as an Overleaf project.

## Upload to Overleaf
1. `bash paper/scripts/package_overleaf.sh` (or `... package_overleaf.sh <venue>` to pre-select a venue, see `venues/`)
2. Overleaf → **New Project → Upload Project** → choose `paper/overleaf_<venue>.zip`
3. Menu → Compiler: **pdfLaTeX**. Main document: `main.tex`.

## Builds
| Venue file | Pages | Blind | Notes |
|---|---|---|---|
| `venue.tex` (default) | 8 | no | full version |
| `venues/isqed2027.tex` | 8 | **yes** | double-blind, PDF metadata scrubbed |
| `venues/sustech2027.tex` | 8 | no | hyperlinks/bookmarks off (SusTech rule) |
| `venues/icmi2027.tex` | 6 | no | `\compacttrue`: short related work, 3 secondary figures/tables dropped |
| `venues/isdfs2027.tex` | 6 | no | same compact build; copyright line ready for camera-ready |
| `venues/isec2027.tex` | 8 | no | full paper (5-8 pp allowed) |
| `venues/anonymous.tex` | 8 | **yes** | generic double-blind |

The 6-page builds fit exactly **once the red `\todo{}` notes are removed**. Keep edits short or check the page count.

## Layout
| Path | What it is |
|---|---|
| `main.tex` | Title, authors, package setup, section includes |
| `venue.tex` | Per-conference switches (blind review, notices). Overwrite with a file from `venues/` |
| `venues/` | One config per target conference + **`CONFERENCES.md`** (deadlines, page limits, calendar) |
| `isec_k12_abstract.tex` | Separate 2-page extended abstract for the ISEC K-12 poster track |
| `sections/` | The paper text, one file per section |
| `figures/` | Generated plots (PDF) + `fig_arch.tex` (TikZ diagram, edit directly) |
| `tables/` | Generated table bodies + `numbers.tex` (every number quoted in the prose) |
| `results/` | Raw JSON from the experiments |
| `scripts/make_figures.py` | Rebuilds every figure and table from `results/` |

## Regenerating results
```bash
cd Model_Code
python hybrid_study.py      # ~25 min on one CPU thread -> hybrid_results.json
python latency_sweep.py     # ~2 min -> latency_results.json
mv hybrid_results.json latency_results.json ../paper/results/
cd ../paper && python scripts/make_figures.py
```
Never type numbers into the prose by hand. Use the macros in `tables/numbers.tex` (e.g. `\mseChainHB`),
so rerunning the experiments updates the whole paper.

## Before submitting
- Search for `\todo{` and resolve every one (they render in red).
- Double-blind venues: set `\anonymoustrue` (the venue files do this), remove the repo URL and acknowledgments,
  and do not upload the README/repo link anywhere in the PDF.
- Run the final PDF through **IEEE PDF eXpress** if the venue requires it (most IEEE conferences do for camera-ready).
