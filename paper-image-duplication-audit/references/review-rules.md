# Review Rules

## Evidence Levels

- `high`: localized candidate score is at least `0.90`, both evidence patches pass minimum size filters, and the highlighted regions show the same internal band/blob structure after contrast normalization.
- `medium`: score is `0.82-0.899` or the visual match is plausible but affected by low contrast, compression, small patch size, or weak surrounding context.
- `low`: score is below `0.82`; use only for exploratory follow-up, not as a reportable concern without additional evidence.

## Western Blot / Gel Checks

Compare only blot-like panels with blot-like panels. Prefer local band-patch candidates over whole-panel similarity because scientific blots often reuse a single lane or band rather than the full panel. Treat same-protein-row matching as the default review boundary; cross-protein similarities are usually exploratory false-positive material unless there is independent evidence of relabeling.

When reviewing a candidate:

1. Confirm both highlighted boxes sit inside blot/gel image content, not labels, legends, axes, or captions.
2. Confirm both evidence patches are large enough to contain real band structure. Treat tiny isolated blobs as low-confidence even when correlation is high.
3. Compare the band shape, speckle pattern, lane boundary, local background, and compression artifacts.
4. Confirm the reported protein-row labels match the highlighted regions. If one row label was propagated from another panel, inspect the panel image and OCR overlay.
5. Check whether the candidate appears in different panels, conditions, lanes, or experimental contexts.
6. Report the figure/panel pair, protein row, page, score, context score, evidence area, and candidate image path.
7. Use cautious wording: "suspicious reuse candidate" or "requires manual review."

## Common False Positives

- Repeated text labels such as cell-line names or molecular weight markers.
- Smooth background windows with little biological signal.
- Tiny dark blobs or single-band fragments that become artificially similar after resizing.
- Reused chart axes, legends, and plot markers.
- Different lanes with simple, low-information bands.
- Images derived from the same control where reuse is disclosed in the caption or methods.

## Reporting Template

Use this compact format:

```text
Finding: Suspicious WB band reuse candidate
Location: Figure <n><panel-a> vs Figure <n><panel-b>, page <page>
Score: <score>, orientation: <orientation>
Context: <context-score>, evidence area: <area-a> px vs <area-b> px
Protein row: <row-label-a> vs <row-label-b>
Evidence: <review image path>
Interpretation: The highlighted blot patches show similar local band/background structure and should be manually checked against the manuscript context.
```
