# Benchmark protocol

ActiveBench compares **camera acquisition policies** under an explicit episode
configuration. A common reconstructor measures the utility of their collected
RGB-D streams. Primary scores come from this common model.

## Episode and inputs

The episode YAML defines the scene, seed, start pose, sensor, motion rates,
collision policy, distractor trajectories and budgets. `prepare_scene.py`
creates portable inputs for any compatible installed scene.

The default generated protocol uses a 640 × 480 camera with 75.17817894° HFOV,
0.5 m/s translation, 60°/s yaw/pitch and navmesh-routed motion. A move costs the
maximum of translation, yaw and pitch times; planning wall time is recorded
separately. Episodes stop at their time/capture budget or an agent's `done()`.
New InteriorGS scenes also default to navmesh routing. The Phase 1 GS
experiments used `collision: none`; those results belong to a separate protocol.

Policies receive RGB, intrinsics, elapsed simulation time and optional depth
and camera pose, as declared by `MethodInfo`. With `stream_observations: true`,
new 1 Hz trajectory frames are delivered at the next decision. Evaluation masks,
held-out reference images and scores are reserved for evaluation. The launcher
also supplies scene bounds/start and, where needed, candidate positions; these
are declared benchmark inputs. `pose_access: none` masks observation/stream
poses; configuration inputs require separate review. See the
[API guide](adding-a-method.md) before making pose-free claims.

MAGICIAN additionally uses reference-surface samples for feasibility. Existing
adapters have different information access and action spaces. Report these
adaptations with results and check protocol compatibility when comparing with
another publication's original experiment.

## Independent measurements

1. **Coverage:** acquired RGB-D visibility of a clean, fixed reference surface.
2. **Common reconstruction:** resample the executed trajectory at 1 Hz and
   1600 × 1200, then train the same gsplat vanilla 3DGS recipe for every policy.
   Default: 30,000 iterations, SH3, RGB L1 + DSSIM, RGB-D initialization,
   dynamic masks disabled, RGB supervision only. The full settings are in each `eval.json`. PSNR/SSIM are recorded per view stratum;
   LPIPS is recorded when its pretrained dependency is available (check
   `config.lpips_available` and `config.lpips_error` before reporting it).
3. **Held-out appearance:** render the common model at fixed evaluation cameras
   and compare against clean targets, using the sampling and PSNR rules below.
4. **Geometry:** Gaussian-center completeness and accuracy relative to sampled
   clean surfaces. These diagnostics measure point proximity. Estimating recovered
   room area requires a surface-based metric. Surface sampling is finite and
   renderer-dependent.

## Scene quality: where to evaluate and why six views

`prepare_evaluation.py` prepares a **fixed cube catalog before acquisition**.
Camera selection uses the clean scene and a separate evaluation seed,
independently of method trajectories, reconstruction scores and distractor masks:

1. Sample 4,000 navigable positions, estimate clearance from nearby reference
   surface samples, then select spatially spread candidates by greedy
   farthest-point sampling: repeatedly choose the point farthest from those
   already selected. This reduces clustering in the sampled navigable space.
2. Place the camera above each candidate floor position. The default attempts
   a floor/ceiling depth-based height adjustment, falling back to 1.4 m if
   the probes or correction are unusable. Clearance is an estimate with
   adaptive fallback; inspect the placement metadata and warnings.
3. Keep a position only if **all six faces** have at least 50% positive reference
   depth; otherwise try the next candidate. Accept 24 positions by default,
   or fail the build if too few pass. This rejects mostly empty reference views.

Each position has front/right/back/left/up/down views with **square 90° FOV**.
These six frusta cover all viewing directions, including the ceiling and floor,
with a method-independent orientation scheme. The default is **24 × 6 = 144
images at 1200 × 1200**. These spatial samples can leave some surfaces and
hidden rooms outside their view. Pixels receive equal weight within each
perspective image.

For each view, compute `PSNR = −10 log10(MSE)` on the **entire RGB image** in
`[0, 1]`, then take the arithmetic mean of the per-image dB values. Each cube
face and each accepted position therefore has equal weight. Scoring includes
all target pixels, including surfaces left unobserved by the method. Targets
show the clean static scene. Evaluation cameras and targets are reserved for
evaluation; independently acquired training views can still overlap them.

The report's archived GS cube catalog uses **1600 × 1600** and a **1.0 m**
fallback height. Its depth/height issue remains unresolved, so those scores
are provisional. Reproduce these scores with the archived catalog. See [Phase 1 protocol](PHASE1_PROTOCOL.md).

## Dynamic impact: paired quality and regional changes

Pair a method's `d0` and `dyn` runs within the same scene/start/seed, configured
budget and reconstruction recipe, and score both on the **same clean targets**.
`ΔQ = Q_dyn − Q_d0` measures the change in reconstruction quality under the
dynamic condition; negative means worse. With the generic cube workflow,
`summarize_benchmark.py` writes this catalog-wide change to
`summary.json → paired_static_dynamic` as `dyn_minus_d0_psnr`.

The Phase 1 report also uses a separate **severe/clean regional catalog**.
Candidate cameras are spatially spread, with some aimed toward distractor
patrols to ensure exposed regions are tested. Labels are fixed from the
scripted dynamic scenario and reused for static runs:

- **Severe:** at least 40% of the image belongs to the union of occlusion masks
  over the 1 Hz capture times. A pixel is flagged when disturbed depth is more
  than 5 mm in front of clean depth. Occlusion may be intermittent, with
  individual frames below 40%.
- **Clean:** zero sampled occlusion, plus at least 0.25 m clearance between
  distractor bounding spheres and sampled visible surfaces during a 10 Hz
  trajectory sweep. The label applies to these finite renderer-based samples.

These masks assign entire images to **view classes**. Compute each class's
full-image PSNR mean, then report:

```text
Δsevere = severe_dyn − severe_d0
Δclean  = clean_dyn  − clean_d0
regional contrast = Δsevere − Δclean
```

A negative contrast means exposed views lost more or improved less than clean
views; both classes may still improve. Keep both static baselines and both
deltas visible. The regional catalog deliberately emphasizes distractor exposure.
Report its PSNR separately from cube scene-quality PSNR. The generic cube summary
provides catalog-wide deltas; the Phase 1 regional contrasts are in
[phase1/pairs.csv](../phase1/pairs.csv).

These pairs measure the **combined acquisition and reconstruction response**.
Distractors can change routes, observations and stopping time as well as pollute
pixels. Report actual duration, frames, path length and coverage alongside PSNR.
To isolate reconstruction contamination, a separate control must replay one
fixed trajectory with clean/disturbed inputs. This control remains future work.

## Comparing and resuming

Use identical scene/config/budget, reference assets and reconstruction settings
across methods within a comparison. Collect several seeds for robustness claims.
`summarize_benchmark.py` reports per-scene seed mean/std, completion counts and
matched static/dynamic deltas. Estimating acquisition variability requires
independent acquisition seeds; per-view variation describes the evaluation views.

The generic runner records expanded inputs, directly referenced asset hashes,
method factory/options/version, platform code and reference recipe. Repeating
the command resumes only matching receipts; changed settings require a new
output root. Preserve the installed dataset distribution, upstream source pins,
weights and environment snapshot as well. Exhaustive provenance for nested
dataset resources and external package contents requires these separate records.

Short budgets, smaller cameras or fewer training iterations are useful pipeline
tests. Record their actual settings and keep them separate from full experiments.
