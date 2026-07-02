# Review Rules

Use these rules before making a written assessment from `report.html`, `report.md`, `results.json`, source figures, OCR overlays, or manually inspected panels. Findings are triage evidence unless raw data, author explanations, or editorial/institutional review supports a stronger conclusion.

## Evidence Levels

- `high`: localized or whole-panel candidate score is at least `0.90`, the compared regions contain meaningful biological/image structure, the visual match survives context review, and the panels represent different stated conditions, samples, time points, or experiments.
- `medium`: score is `0.82-0.899`, or the visual match is plausible but affected by low contrast, compression, small patch size, weak surrounding context, or uncertain labeling.
- `low`: score is below `0.82`, or the observation is qualitative only. Use as exploratory follow-up, not as a reportable concern without additional evidence.

## Anomaly Taxonomy

Classify each concern before reporting it:

- Whole-image duplication: the same panel or photo appears in two places with the same orientation.
- Transformed reuse: the same image appears after mirror, vertical flip, rotation, scaling, cropping, contrast adjustment, or color conversion.
- Local cloning or patching: a region inside one image is copied elsewhere in the same image or another image.
- Undeclared splice or composite: lanes, fields, bands, or objects are assembled from different sources without clear figure boundary markers or legend disclosure.
- Selective enhancement or concealment: local contrast, brightness, erasure, smoothing, or background painting changes the apparent signal.
- Relabeling or recontextualization: an image is reused under a different condition, sample, antibody, exposure, time point, genotype, or disease model.
- Cross-publication reuse: an image appears in another paper, preprint, thesis, poster, or dataset with inconsistent context.
- Chart or plot reuse: the same plot trace, dot cloud, FACS gate, histogram, spectrum, or bar/axis scaffold is reused with different labels.
- Synthetic or AI-like artifact: repeated textures, inconsistent anatomy/cell morphology, impossible labels, or generated-looking objects suggest non-photographic origin. Treat this as a hypothesis requiring extra review, not a standalone conclusion.

## General Review Protocol

1. Identify the figure, panel, page, and visible experimental context before judging the image.
2. Determine whether the concern is automated evidence from the script or a manual/vision-assisted observation.
3. Check captions, methods, supplement legends, and source-data statements for disclosed shared controls, representative images, cropped blots, composite boundaries, or reused reference images.
4. Compare image content after removing labels, axes, legends, and captions from the mental comparison.
5. For transformed reuse, document the transform needed to align the images.
6. For local cloning/splicing, mark the smallest suspicious region and inspect the surrounding background, boundaries, noise, compression, and biological structure.
7. Check whether the alleged reuse changes the scientific claim, such as a different treatment, antibody, cell line, time point, genotype, or sample source.
8. Report only the evidence that is visible and reproducible from the available material. Do not infer intent.

## Western Blot / Gel Checks

Compare only blot-like panels with blot-like panels. Prefer local band-patch candidates over whole-panel similarity because scientific blots often reuse a single lane or band rather than the full panel. Treat same-protein-row matching as the default review boundary; cross-protein similarities are usually exploratory false-positive material unless there is independent evidence of relabeling.

Use evidence aggregates as the second-stage WB review layer. An aggregate should be treated as stronger row-level support only when it contains multiple independent one-to-one local matches, consistent lane offset, consistent orientation, and adequate surrounding-context score. Do not describe an aggregate as a whole-row duplicate unless the full-row diagnostic score and visual row context also support that stronger claim.

When reviewing a WB/gel candidate:

1. Confirm both highlighted boxes sit inside blot/gel image content, not labels, legends, axes, molecular weight markers, or captions.
2. Confirm both evidence patches are large enough to contain real band/background structure. Treat tiny isolated blobs as low-confidence even when correlation is high.
3. Compare band shape, speckle pattern, lane boundary, local background, exposure, and compression artifacts.
4. Confirm the reported protein-row labels match the highlighted regions. If one row label was propagated from another panel, inspect the panel image and OCR overlay.
5. Check whether the candidate appears in different panels, conditions, lanes, antibodies, exposures, or experimental contexts.
6. Check for splice boundaries: vertical discontinuities, mismatched lane widths, abrupt background steps, duplicated lane edges, or missing boundary markers.
7. Report the figure/panel pair, protein row, page, score, context score, evidence area, orientation, and candidate image path.
8. For aggregate evidence, report the independent support count, raw pairwise count, mean-top score, mean context score, dominant orientation, lane-offset consistency, full-row diagnostic score, and aggregate review image.
9. Use cautious wording: "suspicious WB/gel reuse candidate" or "requires manual review."

## Multimodal Verification

Use multimodal verification only as a second-stage reviewer of aggregate evidence images. It can help check whether highlighted boxes visually sit on comparable blot bands and whether the aggregate image supports localized reuse, but it must not replace numerical scoring, OCR/protein-row checks, or human review of the manuscript context. The review can be performed by Codex, OpenClaw, or any other agent/tool that can call a vision-capable model; the audit script only needs to provide the evidence image, metadata, prompt, and result schema.

Keep multimodal verification local to the current agent/session for confidential manuscripts unless sending images to an external or hosted model is approved. When a review is available, record the model or agent name, status, confidence, and rationale in the report; treat `uncertain`, tool errors, or missing review results as non-confirmatory rather than negative evidence.

## Non-WB Whole-Panel Checks

Use `--compare-other-panels` for microscopy, TEM, histology, colony plates, wound-healing images, animal/gross photos, and other raster panels. This mode is intended for whole-panel duplication or transformed reuse, not local cloning.

When reviewing a whole-panel candidate:

1. Confirm both panels are real image panels rather than charts, axes, legends, or text-heavy diagrams.
2. Inspect whether the same biological objects, field layout, edges, dust, debris, scale bars, or background texture align.
3. Check the reported orientation and try the indicated transform mentally or with an overlay if needed.
4. Treat panels with mostly blank background, repeated scale bars, shared axes, or repeated labels as false-positive-prone.
5. If the image was cropped differently, describe the overlapping region rather than claiming full-panel identity.

## Microscopy, Histology, TEM, and Photos

Look for duplicated cell clusters, tissue islands, organelles, lesions, colonies, scratches, bubbles, dust, borders, and background texture. Duplicated representative images can be serious when the caption states different treatment groups, stains, genotypes, animals, fields, or time points.

For suspected local cloning, compare natural noise and biological structure. A clone concern is stronger when both copied signal and copied background/noise repeat together, especially after rotation or scaling. It is weaker when only common round cells or low-information blank background look similar.

## Flow Cytometry, FACS, and Dot Plots

Do not rely on image correlation alone. Compare point-cloud shape, density gradients, gate positions, axis scaling, quadrant labels, event counts, and annotations. Suspicious cases include identical dot clouds with changed labels, reused gates under different conditions, and copied histograms with altered axes.

Report these as chart/plot reuse or relabeling concerns unless raw FCS data or source plots are available.

## Charts, Spectra, and Quantitative Plots

Common axes, legends, marker styles, and templates are not evidence by themselves. Look for identical scatter clouds, line traces, spectra peaks, electropherogram traces, error bars, bar heights, or background grid artifacts under different labels.

If the plot encodes data, request source data or recalculate from the paper when possible. Keep the image-integrity finding separate from any numerical/statistical inconsistency.

## Common False Positives

- Shared controls disclosed in captions or methods.
- Repeated labels, scale bars, molecular weight markers, chart axes, legends, gates, or layout templates.
- Smooth background windows with little biological signal.
- Tiny dark blobs or single-band fragments that become artificially similar after resizing.
- Different lanes with simple, low-information bands.
- Cross-combinations of many similar simple bands within the same protein row. These may create many raw pairwise matches but should not become aggregate findings unless independent one-to-one matches share lane offset, orientation, and context support.
- Images derived from the same control where reuse is disclosed in the caption or methods.
- Adjacent fields from the same specimen where similar tissue architecture is expected.
- Compression artifacts, halftone screens, PDF rasterization, or low-resolution screenshots.
- Stock/example images used as method illustrations and clearly disclosed as such.

## Reporting Template

Use this compact format:

```text
Finding: Suspicious <anomaly type> candidate
Location: Figure <n><panel-a> vs Figure <m><panel-b>, page <page-a>/<page-b>
Evidence type: <WB local band | whole-panel transform | manual clone/splice/plot concern>
Score: <score if available>, orientation: <orientation if available>
Context: <protein row, condition, sample, time point, antibody, or plot label>
Evidence: <review image path or marked source image path>
Interpretation: The highlighted/compared regions show similar <structure/noise/band/cloud/trace> and should be manually checked against the manuscript context and raw source data.
```

Use stronger language only when the image evidence is independently reproducible, contextual differences are clear, and legitimate reuse or disclosed shared controls have been ruled out.
