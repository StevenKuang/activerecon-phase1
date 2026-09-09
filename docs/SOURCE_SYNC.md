# Research and delivery source alignment

The research checkout is `revisit-recon`; the independent delivery repository
is `activerecon-phase1` (Python package: `activebench`). Shared platform changes
are maintained in the research checkout and exported by
`scripts/phase1/export_repo.py`. The delivery has its own clean Git history.

The export includes the generic episode/campaign API, external factories,
scene/reference preparation, method adapters, setup/doctor, fresh-result
summaries, focused platform guides, examples and tests. Its README/package
metadata are sourced from `phase1/project/`; shared guides come directly from
`docs/`, not a second edited copy.

Intentional exclusions are research-only revisit-policy/RL/UAV integrations and
experimental reconstruction backends. The delivery keeps the published Phase 1
adapters and common gsplat backend. It also retains Phase 1 evidence/tools.
Media-production scripts, large data, models and videos are excluded.

Every delivered source entry is recorded in `phase1/source-export.json`, with
the research source path/hash and delivered hash. Selected research-only code
removal produces explicitly different hashes; the common platform files and
guides are byte-identical. Re-export checks import closure. To create a fresh
candidate delivery from the research root:

```bash
python scripts/phase1/export_repo.py --out exports/new-platform-candidate
python scripts/phase1/export_repo.py --check exports/new-platform-candidate
```

Do not hand-copy one modified file into the independent repository and forget
its source. Regenerate the candidate, test it, inspect the diff against the
previous delivery, then update the existing delivery main branch. The
`phase1-v1` and `phase1-v1.1` tags remain unchanged; they identify earlier snapshots.
