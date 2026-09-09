# Benchmark contract

ActiveBench compares **camera acquisition policies** under an explicit episode
configuration. A common reconstructor measures the utility of their collected
RGB-D streams. A planner's internal map is not substituted for that common model.

## Episode and inputs

The episode YAML defines the scene, seed, start pose, sensor, motion rates,
collision policy, distractor trajectories and budgets. `prepare_scene.py`
creates portable inputs for installed scenes; it does not select from the
Phase 1 report roster.

The default generated protocol uses a 640 × 480 camera with 75.17817894° HFOV,
0.5 m/s translation, 60°/s yaw/pitch and navmesh-routed motion. A move costs the
maximum of translation, yaw and pitch times; planning wall time is recorded
separately. Episodes stop at their time/capture budget or an agent's `done()`.
New InteriorGS scenes also default to navmesh routing. The Phase 1 GS
experiments used `collision: none`; those results belong to a separate protocol.

Policies receive RGB, intrinsics, elapsed simulation time and optional depth
and camera pose, as declared by `MethodInfo`. With `stream_observations: true`,
new 1 Hz trajectory frames are delivered at the next decision. Evaluation masks,
held-out reference images and scores are not observation fields. The launcher
also supplies scene bounds/start and, where needed, candidate positions; these
are declared benchmark inputs. `pose_access: none` masks observation/stream
poses, not every possible privileged configuration input. See the
[API guide](adding-a-method.md) before making pose-free claims.

MAGICIAN additionally uses reference-surface samples for feasibility. Existing
adapters do not all have identical information and action spaces. Report these
adaptations with results; adding an adapter alone does not establish a fair
comparison with another publication's original experiment.

## Independent measurements

1. **Coverage:** acquired RGB-D visibility of a clean, fixed reference surface.
2. **Common reconstruction:** resample the executed trajectory at 1 Hz and
   1600 × 1200, then train the same gsplat vanilla 3DGS recipe for every policy.
   Default: 30,000 iterations, SH3, RGB L1 + DSSIM, RGB-D initialization,
   no dynamic mask, no depth loss. The full settings are in each `eval.json`. PSNR/SSIM are recorded per view stratum;
   LPIPS is recorded when its pretrained dependency is available (check
   `config.lpips_available` and `config.lpips_error` before reporting it).
3. **Held-out appearance:** render a fixed clean cube catalog prepared before
   running methods. The default new-campaign catalog contains 24 standing points
   × six square 90° views at 1200 × 1200. Reference positions do not depend on
   method trajectories or dynamic masks. All six views at a point must pass
   the valid-depth threshold; a failed reference build is not a usable catalog.
4. **Geometry:** Gaussian-center completeness and accuracy relative to sampled
   clean surfaces. These are diagnostics, not a physical fraction of recovered
   room geometry. Surface sampling itself is finite and renderer-dependent.

Cube and severe/clean damage-probe PSNR are different metrics. The new generic
pipeline defaults to cube views; the frozen report uses its archived catalogs.
Do not pool them. [PHASE1_PROTOCOL.md](PHASE1_PROTOCOL.md) describes the report's
exact grouping and unresolved GS geometry limitations.

## Comparing and resuming

Use identical scene/config/budget, reference assets and reconstruction settings
across methods within a comparison. Collect several seeds for robustness claims.
`summarize_benchmark.py` reports per-scene seed mean/std, completion counts and
matched static/dynamic deltas. One seed does not estimate variability. Different
methods may visit different routes under distractors; a paired score difference
does not isolate the causal effect of occluded pixels.

The generic runner records expanded inputs, directly referenced asset hashes,
method factory/options/version, platform code and reference recipe. Repeating
the command resumes only matching receipts; changed settings require a new
output root. Preserve the installed dataset distribution, upstream source pins,
weights and environment snapshot as well: nested dataset resources and arbitrary
external Python package contents are not exhaustively hashed by the run receipt.

Short budgets, smaller cameras or fewer training iterations are useful pipeline
tests. Record their actual settings and keep them separate from full experiments.
