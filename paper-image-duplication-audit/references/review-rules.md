# Review Rules

## Evidence Levels

- `high`: localized candidate score is at least `0.90` and the highlighted regions show the same internal band/blob structure after contrast normalization.
- `medium`: score is `0.82-0.899` or the visual match is plausible but affected by low contrast, compression, or small patch size.
- `low`: score is below `0.82`; use only for exploratory follow-up, not as a reportable concern without additional evidence.

## Western Blot / Gel Checks

Compare only blot-like panels with blot-like panels. Prefer local band-patch candidates over whole-panel similarity because scientific blots often reuse a single lane or band rather than the full panel.

When reviewing a candidate:

1. Confirm both highlighted boxes sit inside blot/gel image content, not labels, legends, axes, or captions.
2. Compare the band shape, speckle pattern, lane boundary, local background, and compression artifacts.
3. Check whether the candidate appears in different panels, proteins, conditions, lanes, or experimental contexts.
4. Report the figure/panel pair, page, score, and candidate image path.
5. Use cautious wording: "suspicious reuse candidate" or "requires manual review."

## Common False Positives

- Repeated text labels such as cell-line names or molecular weight markers.
- Smooth background windows with little biological signal.
- Reused chart axes, legends, and plot markers.
- Different lanes with simple, low-information bands.
- Images derived from the same control where reuse is disclosed in the caption or methods.

## Reporting Template

Use this compact format:

```text
Finding: Suspicious WB band reuse candidate
Location: Figure <n><panel-a> vs Figure <n><panel-b>, page <page>
Score: <score>, orientation: <orientation>
Evidence: <review image path>
Interpretation: The highlighted blot patches show similar local band/background structure and should be manually checked against the manuscript context.
```
