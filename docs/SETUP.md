# Install the platform and methods

Use Linux x86-64, conda and an NVIDIA CUDA-capable GPU for simulation and common
3DGS training. The validated machine is an RTX 5090 (32 GB); see
[SYSTEM.md](SYSTEM.md) for CUDA, PyTorch, driver and OS versions. Package locks,
upstream revisions, local adapter patches and checkpoint hashes are versioned
under [phase1/dependencies/](../phase1/dependencies/). That directory is a
reusable dependency snapshot, not a restriction to the Phase 1 scene roster.

## 1. Choose the runtimes you need

| Operation / method | Environment | Additional source / checkpoint |
|---|---|---|
| Mesh simulation, including Habitat test scenes and MP3D | `habitat` | Scene files, navmesh and any dataset config |
| InteriorGS simulation | `habitat-gs` | Patched Habitat-GS source; GS stage + navmesh + non-collidable dataset config |
| Common reconstruction/evaluation | `bencheval` | gsplat; independent of planner environments |
| Random | No extra environment | Runs in the scene simulator |
| R3-RECON (`r3con-pano`) | `r3con` | `R3CON`; compiled panoramic rasterizers |
| MAGICIAN (`magician`) | `magician` | `MAGICIAN`; `weights/macarons/trained_macarons.pth` |
| FisherRF (`fisherrf`) | `fisherrf` | `FisherRF`; its compiled scoring/rasterizer extensions |
| GAVIS (`gavis`) | `gavis` | `gavis`, `gavis-dgr`, `gavis-rasterizer` |
| GLEAM (`gleam`) | `gleam` | `GLEAM`; `ckpt/train_gleam_stage2_wo_gibson_40000000_steps.zip` |

For example: mesh + Random + common reconstruction needs only `habitat` and
`bencheval`. Add `r3con` for R3-RECON, or install all five planner environments
for the full roster. GLEAM's adaptation has been benchmarked on InteriorGS;
its behavior on another scene/dataset requires validation.

Do not merge all methods into one environment: they use incompatible compiled
modules and PyTorch/CUDA versions. The orchestrator launches isolated workers.

## 2. Fetch pinned sources and build environments

Run from this repository's root. Choose **new** source/environment locations;
the setup script refuses to overwrite an existing checkout directory or prefix.
Without `--execute` it prints the commands for review.

```bash
export ACTIVEBENCH_SOURCES="$HOME/activerecon-sources"
export ACTIVEBENCH_ENVS_DIR="$HOME/activerecon-envs"

python scripts/setup.py --sources "$ACTIVEBENCH_SOURCES" --execute
python scripts/setup.py --environment habitat \
  --prefix "$ACTIVEBENCH_ENVS_DIR/habitat" --execute
python scripts/setup.py --environment bencheval \
  --prefix "$ACTIVEBENCH_ENVS_DIR/bencheval" --execute

# Add GS simulation and all five published planners as needed.
for env_name in habitat-gs r3con magician fisherrf gavis gleam; do
  python scripts/setup.py --environment "$env_name" \
    --prefix "$ACTIVEBENCH_ENVS_DIR/$env_name" \
    --sources "$ACTIVEBENCH_SOURCES" --execute || break
done
```

Keep sources separate from the code delivery. Configure their locations before
launching runs so all workers inherit them:

```bash
export HABITAT_SIM_ROOT="$ACTIVEBENCH_SOURCES/habitat-sim"
export HABITAT_GS_ROOT="$ACTIVEBENCH_SOURCES/habitat-gs"
export R3CON_REPO="$ACTIVEBENCH_SOURCES/R3CON"
export MAGICIAN_REPO="$ACTIVEBENCH_SOURCES/MAGICIAN"
export FISHERRF_REPO="$ACTIVEBENCH_SOURCES/FisherRF"
export GAVIS_REPO="$ACTIVEBENCH_SOURCES/gavis"
export GLEAM_REPO="$ACTIVEBENCH_SOURCES/GLEAM"
```

Existing validated installations can be reused. In that case set the variables
to those existing locations and skip creation; environment names still need to
match the table. The default environment root is `~/miniconda3/envs`; external
source defaults are under `~/Projects`. Explicit variables are more portable.

The setup recipe uses Linux conda artifact locks, pinned pip packages and
source-built extensions. A compatible compiler, CUDA toolkit and GPU driver are
required. The evaluator has been rebuilt at a fresh prefix and exercised with
an empty CUDA extension cache. All eight installed environments have passed
import checks, but a fresh rebuild of **all eight on an empty machine** has not
been executed. These are the verified limits, not a universal one-command
installation claim.

## 3. Install weights and simulation data

MAGICIAN's [released weights](https://drive.google.com/drive/folders/1wyc9_QFmcxOz4oerE8kCQ3I8LO5zioZL)
go under `$MAGICIAN_REPO/weights/macarons/trained_macarons.pth`.
Fetch GLEAM's released checkpoint archives with:

```bash
GLEAM_CKPT_DIR="$GLEAM_REPO/ckpt" bash scripts/envs/fetch_gleam_ckpt.sh
```

Use the `stage2_wo_gibson` checkpoint named above. Verify weight bytes against
[weights.json](../phase1/dependencies/weights.json); an uninitialized network
or a different checkpoint is a different method configuration.

Dataset installation and choosing **additional scenes** are covered in
[SCENES.md](SCENES.md). Git does not include licensed datasets, model weights,
scene meshes or GS stages. A source checkout alone does not contain those assets.

## 4. Check your installation and run a short benchmark

```bash
python scripts/doctor.py --environment habitat --environment bencheval
# For all five planners, check all eight runtime environments:
python scripts/doctor.py
# Snapshot actual local hardware/software for your new experiment:
python scripts/phase1/system_info.py --out outputs/system-info.json
```

Doctor checks interpreter paths and imports, not a complete planner run or GPU
training. Follow [RUNNING.md](RUNNING.md) for a real acquisition/reconstruction
smoke test, then increase the budget. Worker failures are reported in per-method
logs. The current validation record is in [ACCEPTANCE.md](ACCEPTANCE.md).

For core development/tests, use an environment containing the core dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## Adapter settings

[configs/methods.yaml](../configs/methods.yaml) supplies the general roster.
FisherRF/GAVIS use their released scoring implementations with a shared local
candidate pool and 800 internal map-training iterations per decision. MAGICIAN
uses beam width/steps 10/10 and the benchmark-prepared reference surface for
feasibility. R3-RECON uses its released panoramic planner. GLEAM uses its released
policy and four turning legs to approximate the original depth ring, with an
observed-occupancy goal gate. These adaptations and early stopping must be
reported with comparisons.

The historical campaign's per-scene exceptions, including reduced-density GS
GAVIS, are confined to `phase1/campaign.json`; the general defaults do not claim
to reproduce those cells. Upstream code retains its original attribution and
licenses; [sources.json](../phase1/dependencies/sources.json) records exact URLs
and revisions.

Spark's browser dependencies are pinned. The optional CPU `build-lod` converter
was validated at Spark 2.1.0; it enables streamed RAD/LoD. PLY is supported when
it is absent. Browser viewing is not a pixel-identity check against gsplat scoring.
