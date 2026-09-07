# Phase 1 release acceptance

Initial release verified on 2026-09-07; instruction validation completed on
2026-09-08, Linux x86-64 and RTX 5090. This release separates
replaying the historical evidence from running a new experiment.

| Check | Executed result | Evidence |
|---|---|---|
| Campaign selection | 64 planned cells, 60 retained models; missing/excluded cells explicit | `phase1/campaign.json` |
| Frozen PSNR verification | All 60 primary and 24 cube model/catalog combinations rendered again; four extra raw-DC GLEAM checks | `phase1/verification/` |
| Report comparison | New CLI checked all 84 PSNR scores and 115 numeric cells in the companion report; zero mismatches | `phase1/acceptance/instructions/` |
| Table regeneration | Four generated files reproduce exactly from frozen JSON | `python scripts/phase1/report.py --check` |
| Exported platform | 234 tests passed, six early-campaign data tests skipped; Python 3.9 compile and wheel installation passed | `phase1/acceptance/tests.xml` |
| Fresh evaluator setup | New conda prefix created by setup.py, empty gsplat JIT build, GS and mesh report scores matched; pip check passed | `phase1/acceptance/instructions/` |
| Scene/method interface | `--scene interior_0007 --method r3con-pano --condition d0` completed the 5 s / 20-iteration pipeline | `phase1/acceptance/instructions/one-run.json` |
| Installed environments | Required imports passed in all eight isolated environments | `phase1/dependencies/runtime-check.json` |
| New acquisition | Six GS adapters and mesh R3-RECON each completed a 5 s smoke episode | `phase1/acceptance/fresh-smoke.json` |
| New common reconstruction | Mesh R3-RECON and GS Random were resampled to 1600 x 1200 and trained for 20 iterations, then scored and exported | `phase1/acceptance/fresh-smoke.json` |
| Portable model re-score | Two actual models and references extracted from the archive scored from the relocated repository, without original-run paths; agreement within 1e-5 dB | `phase1/acceptance/portable-rescore/` |
| Spark | Catalog contains all 60 retained models; browser checks passed for four representative mesh/GS static/dynamic records, playback, stepping, switching and comparison | `phase1/acceptance/spark-browser.json` |
| Report | Updated MD, DOCX, HTML and PDF; 14 figures, 11 tables, no HTML table/document overflow | `phase1/acceptance/report-preview.json` |

The main PSNR tables now use one exported-model color-loading regime. One
MP3D FisherRF score changes by +0.008679 dB; four GLEAM scores change by at most
0.029512 dB in magnitude. Full precision and original values are retained in
`phase1/score-changes.json`. All 17 mesh pairs still have negative regional
contrast, now under the same scoring path. The ten non-GLEAM GS pairs also
retain negative contrast. These corrections do not strengthen the evidence
into a general method ranking or a multi-seed robustness result.

A further table audit corrected GAVIS's mean path change from -1.0% to -0.9%:
the mean of the two full-precision scene percentages is -0.9488263776%.
PSNR and the interpretation are unchanged.

The release did **not** rerun the full 60-cell acquisition and 30,000-step
training matrix, rebuild all eight environments on an empty machine, or
remeasure SSIM, LPIPS and geometry. The initial seven acquisitions and two short reconstructions, plus the
additional tutorial R3-RECON run, validate the integrated execution path;
their scores are excluded from the paper tables. Spark browser testing sampled four
records, while full PSNR verification covered every retained model.

Model and observation archives carry per-file byte counts and SHA-256 hashes.
The evaluation archive was sampled through real extraction and re-scoring;
the full release catalog was tested using identical bytes at relocated
paths. Simulation data and external checkpoints retain their own distribution
requirements and are identified by separate hashes.

For the browser checks, install Playwright and Chromium, start Spark, and run
`node scripts/check_spark_viewer.cjs http://127.0.0.1:8090 outputs/spark-audit`.
By default it visits the full catalog. `SPARK_TEST_SELECTIONS` can restrict
the test to comma-separated `scene__difficulty__seed__method` identifiers;
put two methods from the same scene/condition first for the comparison test.

One damaged working copy of the dynamic InteriorGS R3-RECON model was
detected by CRC/SHA-256 checks. Identical frozen bytes were restored from the
unchanged evaluation archive and both scores passed. The damaged copy was
retained outside the release; no model was retrained or substituted. See
`phase1/acceptance/instructions/artifact-recovery.json`.
