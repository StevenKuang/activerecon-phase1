# Phase 1 supplementary video

**Active Pose-Free 3D Reconstruction with Dynamic Distractors Phase 1 Supplementary**

ActiveBench Visualization · Liming Kuang

Professorship of Photogrammetry and Remote Sensing

Technical University of Munich (TUM)

Both versions contain the same **4:54** film: six static scenes first, then the
five-method MP3D static/dynamic overview. They are silent H.264 MP4s, 30 fps,
with chapter markers.

| Version | Resolution | Size | Download |
|---|---|---|---|
| 1080p | 1920 × 1080 | 287 MB | [MP4](https://github.com/StevenKuang/activerecon-phase1/releases/download/phase1-videos/ActiveBench-Phase1-All-Scenes-1080p.mp4) |
| 4K | 3840 × 2160 | 1.09 GB | [MP4](https://github.com/StevenKuang/activerecon-phase1/releases/download/phase1-videos/ActiveBench-Phase1-All-Scenes-4K.mp4) |

Open the [video release](https://github.com/StevenKuang/activerecon-phase1/releases/tag/phase1-videos)
while signed into an account with access to this private repository. Download a
version and open it in a video player. The README cover links to that release;
it is a download entry point. The videos are stored outside Git, and the media
tag `phase1-videos` is separate from the platform's `main` branch.

Alternatively, with an authenticated GitHub CLI, download and verify 1080p:

```bash
gh release download phase1-videos --repo StevenKuang/activerecon-phase1 \
  --pattern '*1080p.mp4' --pattern SHA256SUMS --dir videos
cd videos
sha256sum --ignore-missing -c SHA256SUMS
```

For 4K, replace `*1080p.mp4` with `*4K.mp4` and use a new output directory.
Checksums are also [versioned in the repository](../phase1/video/SHA256SUMS).

## Chapters

| Time | Content |
|---|---|
| 0:00 | Title and credits |
| 0:08 | Static Scenes |
| 0:11 | interior_0007 |
| 0:51 | interior_0044 |
| 1:31 | apartment_1 |
| 2:11 | van_gogh_room |
| 2:51 | mp3d_17DRP5sb8fy |
| 3:31 | skokloster_castle |
| 4:11 | Static vs. Dynamic |
| 4:14 | MP3D: five methods, static left / dynamic right |

Each scene takes 40 s: 18 s of recorded RGB-D accumulation, a 1 s transition,
a 20 s synchronized final-3DGS camera tour, and a 1 s hold.

## Reading the visualization

- **Cyan:** current camera frustum and viewing direction; the scan inset shows
  actual recorded camera RGB.
- **Amber:** moving distractors replayed at the displayed capture time.
- **Magenta:** contaminated pixels and their accumulated observed points.
- **Final tour:** the camera view of the common Tier-2 3DGS reconstruction,
  with the distractor and contamination overlays removed.

The broader project title includes "Pose-Free"; this Phase 1 benchmark uses
known simulator poses. Static and dynamic acquisitions can follow different
trajectories, so this is a qualitative comparison. InteriorGS GAVIS uses 270 s
and holds its final observation; the other displayed runs use 300 s. Skokloster
shows four methods because the GAVIS acquisition is unavailable. Numerical
results and protocol details are in [Phase 1 results](REPRODUCING.md).

The published files preserve the delivered video bytes. Video-production code
remains in the research workspace, outside the submitted platform source.
