# Target IEEE conferences (researched 2026-10-01)

Shortlist of IEEE Xplore venues that are **in the Bay Area or allow hybrid/remote presentation** and whose deadlines
haven't passed. Details marked *verify* came from a previous edition or an inconsistent website, so check them before
you submit. Every venue requires at least one author to **register and present** for the paper to appear in IEEE Xplore.

## Ranked shortlist

| # | Venue | Dates | Where | Deadline | Pages | Review | System | Build |
|---|---|---|---|---|---|---|---|---|
| 1 | **IEEE ICMI 2027** – Computing & Machine Intelligence | Apr 24–25, 2027 | Mt. Pleasant, MI · **hybrid** (present online) | **Dec 15, 2026** | 5 std, 6 max (+$25) | not stated (*verify*) | CMT | `icmi2027` (6 pp) |
| 2 | **ISQED 2027** – Quality Electronic Design | Apr 14–16, 2027 | **San Francisco** · hybrid | **Oct 25, 2026** (extended, *verify*) | 4–8 (aim 6–8) | **double-blind** | START/Softconf | `isqed2027` (8 pp, anonymous) |
| 3 | **IEEE ISDFS 2027** – Digital Forensics & Security | Mar 25–26, 2027 | **San Jose State** · in person + online | **Jan 31, 2027** | ≤ 6 | single-blind | CMT | `isdfs2027` (6 pp) |
| 4 | **IEEE SusTech 2027** – Technologies for Sustainability | Apr 18–21, 2027 | Portland, OR · **hybrid** | **Nov 1, 2026** (paper) · Feb 16, 2027 (student poster) | 4–8 | not stated | EDAS | `sustech2027` (8 pp, no links) |
| 5 | **IEEE ISEC 2027** – Integrated STEM Education | Mar 13, 2027 | Baltimore · **hybrid** | Jan 10, 2027 (paper) · **Jan 31, 2027 (K-12 poster)** | 5–8 · K-12 poster ≤ 2 | not stated | EDAS | `isec2027` / `isec_k12_abstract.tex` |
| – | IEEE ICAIC 2027 – AI in Cybersecurity (backup) | Mar 3–5, 2027 | Houston · hybrid **contradictory**, email organizers | Dec 30, 2026 | ≤ 6 (+$50) | not stated | CMT | `isdfs2027` style |
| – | IEEE SVCC 2027 – Silicon Valley Cybersecurity (watch) | ~Jun 2027 | San Jose/SF · in person | not announced (2026: Jan 26) | 6–8 / poster 2–3 | double-blind (2025) | EasyChair | `anonymous` |

Checked and ruled out: CCWC 2027 (in-person only, Las Vegas), AIIoT 2027 (in-person), MIT URTC (2026 deadline passed;
high-school authors need a collegiate program), ICCE 2027 (in-person), COMPSAC 2027 (Tokyo, no remote option posted),
IEMCON/UEMCON 2026 (deadlines passed).

## Fit and framing per venue
- **ICMI**: best topical fit (machine intelligence, small/efficient models) and the cheapest remote option
  (~$350 student non-member online). **Recommended first target if you want the strongest chance.**
- **ISQED**: Bay Area, but more selective and industry-heavy. Frame the paper as *design-quality trade-offs*:
  topology vs. on-device latency. $685 author registration (2026 rate), and a student-only registration does not cover a paper.
- **ISDFS**: held locally at SJSU. The call includes AI/ML, but the venue is security-themed. Add a sentence in the intro
  motivating latency for on-device / edge detection models. Copyright line for camera-ready is already in
  `venues/isdfs2027.tex`.
- **SusTech**: frame the paper as *green ML*: compute and latency per unit accuracy. It has its own template (download it)
  and requires PDF/A without hyperlinks or a copyright footer.
- **ISEC**: the only venue that explicitly invites **pre-college authors**. Use `isec_k12_abstract.tex` (a 2-page reflection
  on rigor). A full paper here would need an education angle.

## ⚠️ Duplicate submission
IEEE forbids submitting **the same paper to several venues at the same time**. You can apply to all five, but in sequence:
submit, wait for the decision (or withdraw), revise, then submit to the next. A 2-page K-12 extended abstract (ISEC)
or a student poster abstract (SusTech) is different enough from a full paper that it's usually allowed, but **ask the
chairs** before overlapping.

## Suggested calendar (two realistic chains)

**Chain A – Bay Area first:** ISQED (Oct 25) → if rejected, ISDFS (Jan 31) → SusTech student poster (Feb 16)

**Chain B – best fit and cheapest:** ICMI (Dec 15) → ISEC K-12 poster (Jan 31, separate 2-page abstract) → SusTech
student poster (Feb 16)

| Date | Action |
|---|---|
| **Oct 25, 2026** | ISQED paper (anonymous 8-page build), *or skip to keep ICMI open* |
| Nov 1, 2026 | SusTech paper, only if ISQED was skipped |
| **Dec 15, 2026** | ICMI paper (6-page build), if no other submission is pending |
| Jan 22, 2027 | ISQED notification |
| **Jan 31, 2027** | ISDFS paper (6-page build) **and/or** ISEC K-12 poster abstract |
| Feb 15, 2027 | ISDFS / ICMI notifications |
| **Feb 16, 2027** | SusTech student poster abstract (fallback) |

## Questions to email organizers first
1. Can a minor (under 18) sign the IEEE copyright form, or does a parent or guardian need to co-sign?
2. Does a high-school student qualify for the *student* registration rate?
3. (ICMI) Is review single- or double-blind? Is US Letter IEEEtran OK given the A4 Word template?
4. (ICAIC) Is remote presentation actually allowed?

## Build checklist for every submission
- [ ] `cp venues/<venue>.tex venue.tex` (or `bash scripts/package_overleaf.sh <venue>` → upload zip to Overleaf)
- [ ] Resolve every red `\todo{}` (then they vanish; the 6-page builds only fit with them removed)
- [ ] Double-blind: no name, school, repo link, or acknowledgments; anonymous.4open.science link for code
- [ ] Compile with pdfLaTeX, check page count, check that all fonts are embedded (Overleaf does this by default)
- [ ] Camera-ready only: add the venue's copyright line (`\IEEEpubid`) and pass **IEEE PDF eXpress**
