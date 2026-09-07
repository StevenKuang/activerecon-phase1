# Runtime and external sources

The verified machine uses Linux, an NVIDIA RTX 5090 (32 GiB), separate conda
environments and CUDA-enabled PyTorch. Python 3.9 is used by the mesh simulator
and R3-RECON; GS and GLEAM use Python 3.12. The evaluator and other methods use
Python 3.10. `phase1/dependencies/` records exact conda artifacts, pip versions,
source commits, local source patches and checkpoint SHA-256 hashes.

The code supports Python 3.9-3.12 across isolated processes. A single shared
environment cannot represent the different rasterizer modules used by the
participants. `ACTIVEBENCH_ENVS_DIR` selects the directory containing the
named environments (`habitat`, `habitat-gs`, `bencheval`, `r3con`, `magician`,
`fisherrf`, `gavis`, `gleam`). It defaults to `~/miniconda3/envs`.

Source locations are configurable: `HABITAT_SIM_ROOT`, `HABITAT_GS_ROOT`,
`R3CON_REPO`, `MAGICIAN_REPO`, `FISHERRF_REPO`, `GAVIS_REPO`, `GLEAM_REPO`.
Defaults are checkouts under `~/Projects`. Set these variables before launching
the campaign so the simulator and RPC workers inherit the same locations.

Inspect pinned setup commands first, then execute them against new locations:

```bash
python scripts/phase1/setup.py --sources /new/path/sources
python scripts/phase1/setup.py --sources /new/path/sources --execute
python scripts/phase1/setup.py --environment bencheval --prefix /new/path/envs/bencheval
python scripts/phase1/setup.py --environment bencheval --prefix /new/path/envs/bencheval --execute
# Repeat for each required environment; source-built methods need --sources.
python scripts/phase1/setup.py --environment r3con --prefix /new/path/envs/r3con \
  --sources /new/path/sources --execute
```

The exact conda locks target Linux x86-64. GPU drivers and system/compiler
compatibility still need to match the machine. The recipe is derived from the
verified installed builds; a complete rebuild of all eight environments on an
empty machine has not been executed for this release. Environment checking and
relocated-code smoke results are recorded in the acceptance note.

The `habitat-gs` source patch is required for the locally validated build.
Build CUDA and Bullet support; use a non-collidable dataset configuration for
GS stages, which have no collidable scene mesh. Existing dataset configuration
and every referenced scene/object asset must be available at the expanded
paths in `phase1/configs/`. Keep the exact archived evaluation catalogs.

Obtain simulation data through the dataset authors' distribution: Habitat test
scenes and ReplicaCAD object assets for the mesh group, Matterport3D
17DRP5sb8fy through its access process, and the two InteriorGS stages from
[the GS scene collection](https://huggingface.co/datasets/RukawaY/gs_scenes).
Scene filenames, initial poses and distractor templates are frozen in the
configs. Dataset/checkpoint files are separate from the code repository.
`phase1/dependencies/simulation-assets.json` freezes the actual scene,
navmesh, dataset-config and moving-object bytes. With the two simulator
directories under a common `/your/sources`, verify them using:

```bash
python scripts/phase1/data.py --verify /your/sources \
  --manifest phase1/dependencies/simulation-assets.json
```

MAGICIAN requires `weights/macarons/trained_macarons.pth` from its
[released weights](https://drive.google.com/drive/folders/1wyc9_QFmcxOz4oerE8kCQ3I8LO5zioZL).
GLEAM requires `ckpt/train_gleam_stage2_wo_gibson_40000000_steps.zip` from its
release. `GLEAM_CKPT_DIR=/your/GLEAM/ckpt bash scripts/envs/fetch_gleam_ckpt.sh`
fetches the upstream checkpoint archives. Verify hashes against
`phase1/dependencies/weights.json`; do not
substitute another GLEAM checkpoint or run an uninitialized policy.

Run `python scripts/phase1/doctor.py --out outputs/environment-check.json` to
verify imports in all runtime environments. Then run the short fresh
simulation/reconstruction command in the reproduction guide. `bencheval`
alone suffices for re-scoring saved models. Table regeneration needs only
Python's standard library.

Spark's browser imports are pinned in `webdemo/index.html`. For streamed
SH3/LoD conversion, install the matching Spark `build-lod` executable on PATH.
The verified converter was built from Spark 2.1.0's Rust implementation;
it is an optional CPU conversion step, separate from the browser import.
Viewer inspection is not a pixel-identity check against gsplat evaluation.

## Adapter scope

R3-RECON imports the released incremental voxel/renderability panoramic
planner. MAGICIAN imports occupancy prediction and its multi-step lattice
planning components; its reference-surface feasibility input is documented.
FisherRF and GAVIS import their released scoring code behind the shared local
candidate-pool interface and train an internal map for 800 iterations per
decision. GAVIS's internal model is distinct from the common reconstruction.
GLEAM imports the released policy network/checkpoint and rebuilds observation
mapping against the single ActiveBench camera, using four turning legs to
approximate its original depth ring. Its conservative observed-occupancy goal
gate and deterministic stalls are properties of this adaptation. Random is
the benchmark's short-step random baseline.

External algorithms retain their upstream attribution and licenses. Source
URLs and exact revisions are in `phase1/dependencies/sources.json`. This
repository does not rename those published planners as new methods.
