# Reference hardware and software

The following environment was used for model re-scoring and platform validation
in September 2026. A complete software image was not captured at acquisition
time, so this snapshot describes the reproduction environment.

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5090; 32 GB VRAM (32607 MiB reported) |
| NVIDIA driver | 595.84 |
| CPU | Intel(R) Core(TM) Ultra 9 285K; 24 logical CPUs |
| RAM | 62.17 GiB visible to the OS |
| OS / kernel | Ubuntu 24.04.4 LTS; Linux 7.0.0-28-generic |

| Conda environment | Python | PyTorch | PyTorch CUDA |
|---|---|---|---|
| habitat | 3.9.19 | 2.8.0 | 12.8 |
| habitat-gs | 3.12.13 | 2.12.1 | 13.0 |
| bencheval | 3.10.20 | 2.8.0+cu128 | 12.8 |
| r3con | 3.9.19 | 2.8.0 | 12.8 |
| magician | 3.10.20 | 2.11.0 | 13.0 |
| fisherrf | 3.10.20 | 2.8.0+cu128 | 12.8 |
| gavis | 3.10.20 | 2.8.0+cu128 | 12.8 |
| gleam | 3.12.0 | 2.11.0 | 13.0 |

The common evaluator uses **gsplat 1.5.3** and **NumPy 2.2.6**. Its nvcc compiler is **12.8.93**. The driver advertises CUDA **13.2** compatibility; that is distinct from the CUDA **12.8 / 13.0** runtimes linked by the listed PyTorch builds. Spark browser imports and the optional LoD converter use **2.1.0**.

The Git record contains the [machine snapshot](../phase1/dependencies/reproduction-machine.json), [exact environment locks](../phase1/dependencies/), [upstream commits and patches](../phase1/dependencies/sources.json), and [checkpoint hashes](../phase1/dependencies/weights.json).

To capture your machine without changing installed packages:

```bash
python scripts/phase1/system_info.py --out outputs/system-info.json
# Only the evaluator, if reproducing saved-model results:
python scripts/phase1/system_info.py --environment bencheval --out outputs/evaluator-system.json
```

The [platform validation snapshot](../validation/system-2026-09-09.json)
records the environment used for the 2026-09-09 integration tests.
