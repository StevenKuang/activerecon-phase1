# Phase 1 protocol

The benchmark compares acquisition implementations with known camera pose.
Each planner receives a 640 x 480 camera (75.17817894 degree horizontal FOV),
with depth according to its adapter. Motion costs the maximum of translation,
yaw and pitch time: 0.5 m/s and 60 degrees/s. Planning wall time is recorded
separately; the simulated clock charges motion time only.

Reconstruction input is sampled at 1 Hz, including time zero, and re-rendered
at 1600 x 1200 using the recorded trajectory. Planners receive newly recorded
640 x 480 stream observations at their next update. Sampling rate and duration
determine the reconstruction frame count independently of decision count.

| Setting | Mesh | InteriorGS |
|---|---|---|
| Scenes | Apartment, Van Gogh, MP3D 17DRP5sb8fy, Skokloster | interior_0007, interior_0044 |
| Motion execution | Navmesh-routed | Unrouted, collision none |
| Shared evaluation | 20 severe + 20 clean; MP3D 19 + 20 | 40 severe + 40 clean |
| Planned/retained reconstructions | 40 / 36 | 24 / 24 |
| Methods | Random, R3-RECON, MAGICIAN, FisherRF, GAVIS | Those five plus GLEAM |
| GAVIS | 300 s, planner initialization 10,000 points/view | 270 s, 2,000 points/view |
| Other budgets | 300 s | 300 s; dynamic GLEAM ends at 244.774 / 255.570 s |
| Distractors | Four to six mixed objects, 0.189-0.733 m/s | Six chairs, 0.35 m/s |

Common reconstruction: 30,000 iterations, seed 0, SH3, 10,000 initial RGB-D
points per frame, RGB L1 + DSSIM supervision only, dynamic masks disabled,
densification gradient threshold 0.0002, and uncapped Gaussian count. The altered-recipe
Van Gogh dynamic Random/FisherRF retries are excluded. Missing Skokloster
GAVIS acquisitions are kept as missing entries.

All Phase 1 PSNR tables use saved models, reconstructing the DC
band from stored RGB and retaining the higher SH bands. Original scores and
the exact differences remain in the evidence. This standardizes the loading
regime of older and newer model exports; coefficients clipped in older artifacts
remain clipped. Rebuilding a new model can give a different
score and is recorded as a new run.

## Evaluation catalogs and dynamic deltas

The report's **shared PSNR** is the per-image dB mean over the severe/clean
catalog in the table above, at 1600 × 1200. For MP3D, the overall mean weights
the 19 severe and 20 clean views by their counts.
Labels come from the dynamic scenario and are reused for static runs. PSNR
uses every RGB pixel in each clean-target image.

The separate **provisional GS cube PSNR** uses 24 spatially spread standing
points × six square 90° views at 1600 × 1600, with evaluation seed 20260727.
Selection uses 4,000 navmesh samples, estimated surface clearance and
farthest-point spreading; every face must have at least 50% positive depth.
Camera height uses a depth-based adjustment with a 1.0 m fallback. Current
`prepare_evaluation.py` defaults to 1200 × 1200 and a 1.4 m fallback;
restore archived cameras/targets to reproduce the report. The cube catalog
provides a single spatial score for each GS reconstruction; mesh tables use
their shared severe/clean catalogs.

The [sampling rationale and formulas](PROTOCOL.md#scene-quality-where-to-evaluate-and-why-six-views)
explain why six directions are used, how PSNR is averaged, and how regional
labels are selected. The GS cube set's unresolved downward-depth and camera-height
issue limits geometric and floor/ceiling interpretations. The selected GS
assets have zero higher-order SH signal, so appearance findings apply to
direction-independent Gaussian color. Gaussian-center completeness measures
point proximity; estimating recovered room area requires a surface-based metric.

Dynamic-minus-static severe and clean deltas are paired within scene, method
and seed in `phase1/pairs.csv`. Regional contrast is `Δsevere − Δclean`;
negative means more loss or less gain in severe views. For example, GS FisherRF
has mean `Δsevere = −0.56 dB`, `Δclean = +1.83 dB`, yet its overall shared PSNR
improves by `+0.64 dB`: the aggregate conceals the regional loss. All means use
full-precision per-scene deltas before rounding. Trajectories may change, and two
GLEAM dynamic durations are shorter. These single-seed results describe the
combined acquisition/reconstruction response. Isolating occlusion effects,
estimating statistical variability and validating collision-free flight require
additional experiments. MAGICIAN's adapter uses a reference surface in feasibility
checks; information access and action spaces vary across implementations.
