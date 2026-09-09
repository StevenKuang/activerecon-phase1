# Datasets and additional scenes

The platform accepts installed Habitat-compatible stage files or scene-dataset
handles. Add compatible scenes through episode configurations. The six
Phase 1 scenes provide examples of this workflow.

| Dataset | Inputs | Simulator |
|---|---|---|
| Habitat test scenes | `.glb`, accompanying navmesh; default dataset config | `habitat` |
| Matterport3D (MP3D) | Downloaded scene `.glb` + navmesh; supplied dataset config where applicable | `habitat` |
| InteriorGS | `.gs.ply` + navmesh + GS dataset config with non-collidable stages | `habitat-gs` |
| ReplicaCAD | Dataset config, `apt_0` etc. scene handles, navmeshes and referenced assets | `habitat` |

MP3D acquisition follows the dataset's official access/terms process. InteriorGS
assets are distributed in the [GS scene collection](https://huggingface.co/datasets/RukawaY/gs_scenes).
Use the supplied Habitat asset downloader for public test scenes after simulator
installation:

```bash
conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/habitat" \
  python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes \
  --data-path "$HABITAT_SIM_ROOT/data"
```

Record dataset versions alongside each experiment. Moving object templates
are an additional requirement for dynamic scenes.

## Discover installed scenes

```bash
python scripts/list_scenes.py --data-root "$HABITAT_SIM_ROOT/data"
python scripts/list_scenes.py --data-root "$HABITAT_SIM_ROOT/data" --datasets mp3d
python scripts/list_scenes.py --data-root "$HABITAT_GS_ROOT/data" --datasets interiorgs
```

The editable catalog is [configs/datasets.yaml](../configs/datasets.yaml).
Discovery scans the installed datasets.
Explicit catalog entries describe expected paths; check installation before
using them. Change a dataset root/glob for another split or directory layout.
You can also skip discovery and pass any installed scene directly.

## Prepare an additional MP3D or mesh scene

```bash
conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/habitat" \
  python scripts/prepare_scene.py \
  --scene-path /datasets/mp3d/OTHER_ID/OTHER_ID.glb \
  --name mp3d_OTHER_ID --out-dir outputs/new-scenes/configs --seeds 0 1 2
```

This creates `mp3d_OTHER_ID__d0__s{0,1,2}.yaml` with a shared render-checked start
and 300 s budgets. Pass `--dataset-config /path/to/dataset.json` when the asset
requires one. For ReplicaCAD, pass `--scene-path apt_1` plus its dataset config.
Automatic scene/reference preparation requires the scene asset and a loadable
navmesh.

## Prepare an additional InteriorGS scene

Create the non-collidable dataset config once per installed split. Gaussian
stages use Gaussian geometry; this setting disables stage-mesh collision
construction while permitting Bullet-backed distractors.

```bash
python scripts/gs_make_noncollide_config.py \
  --gs-dir "$HABITAT_GS_ROOT/data/scene_datasets/gs_scenes" --splits train

conda run --no-capture-output -p "$ACTIVEBENCH_ENVS_DIR/habitat-gs" \
  python scripts/prepare_scene.py \
  --scene-path "$HABITAT_GS_ROOT/data/scene_datasets/gs_scenes/train/interior_0022_840117/interior_0022_840117.gs.ply" \
  --dataset-config "$HABITAT_GS_ROOT/data/scene_datasets/gs_scenes/train_activebench_noncollide.scene_dataset_config.json" \
  --name interior_0022 --out-dir outputs/new-gs/configs --seeds 0 1 2
```

Substitute another installed stage and use the matching split config.
Preparation defaults to navmesh motion on both backends. Choosing
`--collision none` changes the protocol and must be reported separately.

## Add dynamic distractors

Append these options to `prepare_scene.py`:

```bash
--conditions d0 dyn \
--object-template /datasets/objects/chair.object_config.json \
--object-diag-m 1.1 --object-center-height 0.5 --object-scale 1.0 --distractors 3
```

The template origin/scale must match the object asset; diagonal is a conservative
**scaled** size used for route clearance. Inspect the resulting placement before
reporting experiments. Routes follow navmesh shortest-path polylines on the start
island, with sampled clearance checks; the build requires the requested number
of valid routes. Full swept-volume collision validation requires additional
checks. Static and dynamic conditions share camera/start/budget and seed.

Generated YAML is the editable interface: adjust motion, budget, task or object
trajectories there and use a new output root. `start_pose` yaw/pitch are radians;
HFOV and motion rates with `_deg` are degrees. Keep unique scene names within
one experiment. Reserve `__` for the generated field separator.

Next, prepare clean references and run methods using [RUNNING.md](RUNNING.md).
References use only the `d0` configs, so keep a clean config even if you later
select only `--conditions dyn`. Mesh and GS preparations run in their respective
environments; a general campaign can launch both backends, but aggregate only
compatible measurement protocols together.
