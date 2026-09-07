# ActiveBench

ActiveBench evaluates active camera acquisition through a common RGB-D
simulation interface, fixed-rate trajectory sampling, a common gsplat
reconstructor and Spark replay. This repository contains the Phase 1 benchmark
and its frozen evidence.

The campaign has two protocol groups: four mesh scenes and two InteriorGS
scenes. It retains 60 standard reconstruction results from 64 planned cells.
The two groups have different motion, evaluation-camera and disturbance
protocols; their scores are reported separately. The evaluated implementations
are R3-RECON, MAGICIAN, FisherRF, GAVIS, GLEAM and Random. GLEAM was evaluated
only on GS scenes. GAVIS is a reduced-density 270 s reference on GS scenes.

Start with [the protocol](docs/PROTOCOL.md), [reproduction instructions](docs/REPRODUCING.md),
[generated results](phase1/RESULTS.md) and [method setup](docs/SETUP.md).
See [the acceptance record](docs/ACCEPTANCE.md) for the executed checks and
the limits of the reproduction claims.

Regenerate and verify the reported tables without a GPU or datasets:

```bash
python scripts/phase1/report.py --check
python scripts/phase1/report.py
```

Inspect the complete acquisition/reconstruction plan before running it:

```bash
python scripts/phase1/run.py
python scripts/phase1/run.py --group gs --execute
```

The default plan runs the 60 retained standard cells. `--include-missing`
attempts all 64 planned cells, keeping any new outcomes separate from the
frozen evidence. [The campaign manifest](phase1/campaign.json) declares every
budget, method option, completion state and historical source.

Spark is the demonstration frontend. After restoring the evaluation, uniform
stream and viewer-assets archives into `data/phase1/`:

```bash
python scripts/phase1/view.py --port 8090
```

Open http://127.0.0.1:8090. Select a recording, click **Apply changes**, and use
playback, stepping and the timeline. Enable **compare (split view)** to compare
methods or static/dynamic conditions with a shared camera and clock. The
viewer displays final stored reconstructions; playback advances recorded
acquisition. Install Spark `build-lod` for streamed RAD/LoD models; PLY is the
supported fallback when the converter is absent.

Run the platform tests in the simulator/core environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

Large model/observation archives and simulation assets stay outside Git.
The source tree excludes revisit-policy, RL/UAV and pose-free reconstruction
experiments. [Source export hashes](phase1/source-export.json) and
[dependency provenance](phase1/dependencies/sources.json) identify the code
and local upstream patches used for this release.
