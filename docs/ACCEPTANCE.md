# Validation and known limits

Validation covers the benchmark interfaces, short simulation/training runs,
viewer playback and reproduction of the saved Phase 1 PSNR scores. Tests ran
on Linux with an RTX 5090; [SYSTEM.md](SYSTEM.md) lists the software versions.

## Tested configurations

| Component | Coverage | Record |
|---|---|---|
| Platform tests | 246 passed; six replay tests skipped because their run data was unavailable | [Integration record](../validation/platform-2026-09-09.json) |
| Short tutorial | Van Gogh, Random, 6 s; acquisition, 20-step common training, 12 evaluation views and summary; resume checked | [Integration record](../validation/platform-2026-09-09.json) |
| Additional scene and method API | InteriorGS `interior_0022_840117`, Random and the external spin factory, both static/dynamic; four complete short runs using RPC and common training | [Integration record](../validation/platform-2026-09-09.json) |
| Method adapters | Random, R3-RECON, MAGICIAN, FisherRF, GAVIS and GLEAM each completed a 12 s GS acquisition through the general runner | [Integration record](../validation/platform-2026-09-09.json) |
| Runtime setup | All eight environments passed import checks; a fresh evaluator environment and empty CUDA extension cache were tested with GS and mesh models | [Imports](../phase1/dependencies/runtime-check.json) · [Evaluator setup](../phase1/acceptance/instructions/) |
| Phase 1 PSNR | 60 retained models, 84 model/catalog scores; saved tables regenerate exactly and all 115 numerical report cells match | [Model re-scores](../phase1/verification/) · [Table checks](../phase1/acceptance/instructions/) |
| Portable data | Two models and their references were extracted and scored at relocated paths, agreeing within 0.00001 dB | [Re-scoring record](../phase1/acceptance/portable-rescore/) |
| Viewer | Playback, stepping, selection and split comparison checked in a browser on four Phase 1 records; four additional-scene model/replay pairs prepared and served over HTTP | [Browser checks](../phase1/acceptance/spark-browser.json) · [Integration record](../validation/platform-2026-09-09.json) |

## Limits

Short runs check integration; they do not establish reconstruction quality or
full-budget performance. Common training was checked separately from the
six-method acquisition test. A fresh full-budget campaign and a clean rebuild
of all eight environments have not been tested. Additional MP3D scenes use
the same scene interface, but only the Phase 1 MP3D asset has been exercised.

Model re-scoring recomputes PSNR. SSIM, LPIPS and geometry in the Phase 1 tables
use recorded measurements. The provisional GS cube catalog retains its
unresolved depth/height issue. See [PHASE1_PROTOCOL.md](PHASE1_PROTOCOL.md) for
budget exceptions and limits on cross-method and dynamic-impact conclusions.

## Run checks locally

From an environment with the core development dependencies installed:

```bash
python -m pytest
python scripts/phase1/report.py --check
```

Then follow the [short tutorial](RUNNING.md#a-short-real-end-to-end-run) to check
simulation, training and evaluation on your installation.

For browser checks, install Playwright and Chromium, start the viewer, and run:

```bash
node scripts/check_spark_viewer.cjs http://127.0.0.1:8090 outputs/spark-audit
```

The browser check visits the full catalog by default. Set
`SPARK_TEST_SELECTIONS` to comma-separated `scene__difficulty__seed__method`
identifiers to select a subset; put two methods from the same scene/condition
first for the comparison check.
