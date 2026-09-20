# SatQuery AI — Backend

FastAPI service: orchestrator, task classification, specialist services
(VQA/captioning, change detection, optical+SAR fusion), and the API. See the
repo root [README.md](../README.md) for setup, deployment, and the
project-wide status table.

## Evaluation

All numbers below come from running the actual orchestrator end-to-end
(`app.services.orchestrator.run_analysis()`) against real, officially-sourced
benchmark data — not a bypassed direct model call — via the scripts in
`backend/scripts/evaluate_*.py`. Raw per-sample results are in
`backend/evaluation_results/` and `backend/data/<dataset>/`.

**BigEarthNet.txt held-out test split** (same-domain generalization; never
seen during LoRA fine-tuning): binary 63.3%, MCQ 73.2%, overall binary+MCQ
68.0% (n=150). Captioning reported qualitatively (see Limitations).

**RSVQA-LR** (`dmarsili/RSVQA-LR-2k`, verified faithful to the official
Zenodo release 6344334 — exact question text, answer values, and one image
checked pixel-for-pixel): overall 50.0% (n=150). The `count` category
(7.1%) was specifically investigated and confirmed to be a genuine, expected
benchmark property, not a bug or data issue — RSVQA-LR's count ground truth
is derived from underlying vector/GIS data at finer granularity than the
displayed low-resolution image, so large counts (in the hundreds) are often
not verifiable by eye even though they're the real official answer.

**VRSBench** (`xiang709/VRSBench`, the authors' own HuggingFace release,
matching GitHub repo `lx709/VRSBench`) — VQA and grounding subsets sampled
at reduced size due to a documented hosting constraint (see note below);
captioning could not be corpus-scored at all for the same reason.

- *VQA* (n=50, real orchestrator, `task_hint="vqa"`): overall 28.0% exact-match.
  Per-category: object existence 55.6%, object quantity 50.0%, object color
  20.0%, scene type 16.7%, object position 12.5%, object category/reasoning/
  direction 0% (n≤6 each, too small to read as a real 0%). This is a harder,
  more diverse benchmark than BigEarthNet.txt, so a lower score than the
  held-out number is expected, not anomalous.
- *Captioning* — **not corpus-scored**: a single real sample through the
  orchestrator took 62 seconds (non-batched, no early stopping, ~216-word
  output near the 256-token cap), making a representative sample size
  impractical in this session (3 real complete samples were captured before
  reducing scope; see `evaluation_results/vrsbench_captioning_eval_results.json`).
  Reported qualitatively instead, and it's the same failure mode already
  documented below for BigEarthNet — fabricating unrelated specifics rather
  than describing the actual scene:
  - Ground truth: *"...a region with two expressway service areas...
    surrounded by vegetation."* → Prediction: *"...a large, expansive area
    of agricultural land... a mixed-use development... The overall climate
    is warm..."* — misses the expressway service areas entirely.
  - Ground truth: *"...an airport... a densely packed suburban area with
    houses, trees, and winding roads..."* → Prediction: *"The dominant
    feature is the 'Schoenbrunn' reservoir... bordered by the 'Klagenfurt'
    and 'Kirche' municipalities..."* — invents specific European place
    names on an image that isn't European at all, the exact BigEarthNet.txt
    leakage pattern the Limitations section below describes, now confirmed
    on a second, independent benchmark.
- *Grounding* (n=25, real orchestrator through `grounding_service.py`,
  including the oversized-box size guard): Precision@IoU=0.5 = 20.0%, mean
  IoU = 0.162. The grounding guard fired on 7/25 samples (28%) — notably
  higher than the fire rate observed on our own app's short category-style
  phrases, because VRSBench's referring expressions are full descriptive
  sentences (e.g. *"The small docked ship positioned towards the middle-
  right part of the image, surrounded by water."*), which push Grounding
  DINO tiny toward the same oversized/degenerate-box failure mode
  documented in the grounding limitations below. Some individual samples
  localize very well (IoU=0.80 on one ship referring expression); most miss
  (IoU=0), consistent with this detector's known weakness on long natural-
  language phrases rather than short category names.

*Hosting note:* VRSBench's official image archive (`Images_val.zip`, ~4GB
on HuggingFace) exhibited severe, inconsistent server-side throttling in
this environment — both `huggingface_hub`'s Python client and a plain
`curl` of the resolved URL degraded from ~7MB/s to under 250KB/s
mid-download across repeated attempts. Rather than substitute an
unverified mirror, sample sizes for VQA and grounding were reduced (50 and
25 respectively, from a target of 100-150) to fit within a practical time
budget, using range-request extraction of only the needed images from the
official archive. All data used is still the authors' own official
release, just a smaller slice of it.

**CDVQA** (change-VQA; official annotations from the authors' own GitHub
repo `YZHJessica/CDVQA`, images from the official SECOND semantic-change-
detection dataset at `captain-whu.github.io/SCD` — cross-referenced and
confirmed CDVQA's Val split draws from SECOND's *train* image pool, not
SECOND's own test split): **0.0% exact-match across all 200 samples and
all 8 question types.**

This was investigated before being reported, the same way the RSVQA count
anomaly was — and unlike that case, this one is a real, structural
capability gap, not a benchmark-format quirk:

- CDVQA's ground truth is always a short token: a bare "yes"/"no", a
  semantic land-cover class name (e.g. `NVG_surface`, `low_vegetation`), or
  a bucketed ratio label (e.g. `10_to_20`) — because CDVQA tests **semantic**
  (multi-class) change detection, sourced from SECOND's 6-class land-cover
  labels.
- Our `change_vqa` path is a **binary** changed/unchanged detector by
  deliberate design (see the Grounding/change limitations below and
  `change_service.py`'s own docstring: "a binary change mask has no
  semantic label for building/road/vegetation/etc."). Its answer is always
  a full descriptive sentence about percent-changed and coarse location —
  never a bare yes/no, class name, or ratio bucket — so exact-match against
  CDVQA's ground truth format is failing by construction, independent of
  whether the underlying detection is qualitatively reasonable.
- Example: *"Did the areas of non-vegetated ground surface change?"*
  (GT=`yes`) got *"A low level of change was detected... affecting
  approximately 2.09%... concentrated in the top-right (56%) and
  bottom-right (44%)."* — a real, computed answer about the binary mask,
  just never phrased as the yes/no CDVQA expects, and with no way to
  isolate "non-vegetated ground surface" specifically from a binary mask.
- Secondary, smaller factor: the OOD guard originally fired on only 9.7% of
  samples despite the underlying detector clearly failing on a much larger
  share — two hand-verified examples (a new warehouse and a new industrial
  building, both large enough to be obvious by eye) were completely missed
  by the detector (0.0% and 12.35% reported vs. a true magnitude of roughly
  20-50%) while scoring z=2.76-2.77, just under the guard's z>3.0 bar.
  Investigated per the same evidence-first standard as the count/format
  findings above (see `ood_guard.py`'s `NEAR_ZERO_CHANGE_Z_THRESHOLD`): a
  second, narrower check now flags when the detector reports ~no change at
  all *and* the pixel z-score is still elevated (>2.0, below the primary
  3.0 bar). Calibrated against 100 real in-distribution LEVIR-CD pairs (2.0%
  false-positive rate there) and confirmed against the live code on both
  that control set and the same 200 CDVQA samples: **coverage roughly
  doubled, 9.7% → 18.8%**, surfaced as a distinctly-named
  `ood_guard_near_zero_change` trace step so it's never confused with the
  original check's different (and differently-calibrated) false-positive
  profile. This narrows the blind spot; it doesn't close it — detections
  that are nonzero but still severely under-estimated (like the 12.35%
  case above) fall outside this specific check by design, since it only
  triggers on near-zero output.

No changes were made to the underlying detector itself in either
investigation — both are honesty-guard widening, not accuracy fixes.
Closing the CDVQA format gap for real would require either a semantic
(multi-class) change detector, or new query-parsing
logic to map CDVQA-style questions onto whatever a binary detector can
actually answer — both real engineering work, not a config change.

## Limitations

**VQA/captioning model (SmolVLM-500M-Instruct + LoRA fine-tuned on
BigEarthNet.txt):**

- Direct factual VQA questions (yes/no, multiple-choice, "is there a river",
  "what land-cover class is present") are reliably grounded in the actual
  image — this is what the held-out accuracy numbers reflect.
- Open-ended, long-form description ("describe this image in detail") can
  still fabricate plausible-but-wrong specifics — a country, a climate zone,
  a land-cover area figure — especially on complex or out-of-distribution
  scenes (dense cities, landmarks, anything outside Sentinel-2 European
  land-cover imagery). It can also miss the single most salient feature of
  the scene (e.g. a river) while confidently describing something else.
- This is a known constraint of the current adapter, not a pipeline or
  routing bug: the LoRA was trained on a narrow, Europe-only sample of
  ~3,600 examples, and a rank-8 adapter on a 500M model has limited capacity
  to override that training distribution when faced with genuinely novel
  input. It would take a larger/more diverse training set (or a bigger base
  model) to close, not a prompt or routing change.

**Grounding / object counting (Grounding DINO tiny, zero-shot):**

Tested directly on 4 real remote-sensing images (not assumed from the
paper's natural-image benchmarks) for "building" and "water body". Results
are sharply bimodal by scene type, not a uniform pass or fail:

| Image (scene type) | "building" | "water body" |
|---|---|---|
| Container port (industrial) | Fail -- single box covers 83% of the image (the whole container yard); no real building isolated | Pass -- tight box on the actual visible water (score 0.81) |
| Dense urban plaza (Delhi) | Partial -- one box correctly lands on a real building cluster (score 0.31) but groups ~5-10 of the dozens of visible rooftops instead of finding them individually; a second box is a degenerate full-image false positive (score 0.45) | Fail (false positive) -- confidently (score 0.46) boxed the green park/roundabout as a water body; there is no water anywhere in this image |
| Rural houses + 2 ponds | Pass -- 40 individually tight, correctly localized boxes, closely matching the ~35-40 visible rooftops | Pass -- both ponds found with tight, accurate boxes (scores 0.77, 0.33) |
| Wide-area river estuary (low zoom) | Fail -- single box covers 99% of the image, no localization | Fail -- single box covers 99% of the image; a real, clearly-shaped river is visible but the box gives no usable location despite a high score (0.86) |

Net: 3/8 correct, 1/8 partially useful, 4/8 failed, including one confident
false positive on an object that isn't present at all. The failure mode
tracks scene density/zoom, not the specific phrase: moderate-density scenes
with individually-resolvable objects (the rural image) detect well; dense
industrial/urban scenes and very-high-altitude/wide-area imagery both tend
to collapse to a single near-full-image box, occasionally with high
confidence on the wrong content entirely.

This is concrete, evidenced justification -- not a generic "trained on
natural photos" caveat -- for revisiting object detection with **LAE-DINO**
(Pan et al., AAAI 2025), a detector trained specifically on remote-sensing
imagery, as a dedicated follow-up. That swap was investigated and deferred
for now: LAE-DINO ships only as raw mmdetection/mmcv checkpoints requiring
Python 3.8 + torch 1.10.0+cu113 with compiled CUDA ops, which conflicts
with this project's Python 3.12 / torch 2.5.1+cu121 environment -- it would
need a separate isolated service, not a drop-in replacement.

*Size guard (added after this evaluation):* `grounding_service.py` now
drops any box covering more than 75% of the image before returning a
result, since that shape of failure (a single box spanning most of the
frame) accounted for most of the failures above. Re-tested against the
same 4 images: the port-building case (was an 83% box) now returns an
explicit "did not localize confidently" message instead of a wrong count;
the two river-estuary cases improved from a useless 99%-area box to a
smaller, partially-plausible one. This guard does **not** fix the Delhi
false positive (park boxed as "water body") -- that box was only 29% of
the image area, nowhere near the oversized threshold, because the failure
there was wrong content at a normal size, not an oversized box. An
area-based check cannot detect that failure mode by construction; it
would need a semantic check or a better detector (see LAE-DINO above).
