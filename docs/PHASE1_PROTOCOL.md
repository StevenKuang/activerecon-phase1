# Phase 1 protocol

The benchmark compares acquisition implementations with known camera pose.
Each planner receives a 640 x 480 camera (75.17817894 degree horizontal FOV),
with depth according to its adapter. Motion costs the maximum of translation,
yaw and pitch time: 0.5 m/s and 60 degrees/s. Planning wall time is recorded
separately and does not advance the motion clock.

Reconstruction input is sampled at 1 Hz, including time zero, and re-rendered
at 1600 x 1200 without re-running planning. Planners receive newly recorded
640 x 480 stream observations at their next update. Decision count does not
increase the uniform-time reconstruction frame count.

| Setting | Mesh | InteriorGS |
|---|---|---|
| Scenes | Apartment, Van Gogh, MP3D 17DRP5sb8fy, Skokloster | interior_0007, interior_0044 |
| Motion execution | Navmesh-routed | Unrouted, collision none |
| Shared evaluation | 20 severe + 20 clean; MP3D 19 + 20 | 40 severe + 40 clean |
| Planned/retained reconstructions | 40 / 36 | 24 / 24 |
| Methods | Five; no GLEAM | Six, including GLEAM |
| GAVIS | 300 s, planner initialization 10,000 points/view | 270 s, 2,000 points/view |
| Other budgets | 300 s | 300 s; dynamic GLEAM ends at 244.774 / 255.570 s |
| Distractors | Four to six mixed objects, 0.189-0.733 m/s | Six chairs, 0.35 m/s |

Common reconstruction: 30,000 iterations, seed 0, SH3, 10,000 initial RGB-D
points per frame, RGB L1 + DSSIM, no depth loss, no dynamic mask, densification
gradient threshold 0.0002, no Gaussian-count cap. Historical altered-recipe
Van Gogh dynamic Random/FisherRF retries are excluded. Missing Skokloster
GAVIS acquisitions are kept as missing entries.

All release PSNR tables use reloaded exported models, reconstructing the DC
band from stored RGB and retaining the higher SH bands. Original scores and
the exact differences remain in the evidence. This standardizes the loading
regime of older and newer model exports; it does not restore coefficients
already clipped in old artifacts. Rebuilding a new model can give a different
score and is recorded as a new run.

The provisional GS cube set has 24 standing points and six views each,
1600 x 1600 and 90 degree FOV. Its unresolved downward-depth and camera-height
issue limits geometric and floor/ceiling interpretations. The selected GS
assets have zero higher-order SH signal: this is not evidence for recovering
view-dependent reflectance. Gaussian-center completeness is a diagnostic,
not a physical fraction of room surface recovered.

Dynamic-minus-static severe and clean deltas are paired within scene, method
and seed. Negative severe-minus-clean contrast is relative regional damage;
severe PSNR need not decrease absolutely. Trajectories may change, and two
GLEAM dynamic durations are shorter. These single-seed results do not isolate
direct occlusion effects, establish significance or validate collision-free
flight. MAGICIAN's adapter uses a reference surface in feasibility checks;
the implementations do not share an identical information/action contract.
