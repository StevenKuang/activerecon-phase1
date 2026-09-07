"""Title-free report figures from the release tables (no new measurements)."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "phase1")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/phase1-figures")
    args = ap.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import ScalarFormatter, NullLocator
    rows = list(csv.DictReader((args.data / "results.csv").open()))
    pairs = list(csv.DictReader((args.data / "pairs.csv").open()))
    args.out.mkdir(parents=True, exist_ok=True)
    methods = ["r3con-pano", "magician", "fisherrf", "gleam", "random", "gavis"]
    names = dict(zip(methods, ["R3-RECON", "MAGICIAN", "FisherRF", "GLEAM", "Random", "GAVIS"]))
    colors = dict(zip(methods, ["#087F8C", "#3169B0", "#CC7028", "#8263AF", "#5E6875", "#AD5269"]))
    severe, clean, ink = "#C3652D", "#087F8C", "#182D45"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "svg.fonttype": "none", "axes.unicode_minus": True})

    def style(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(color="#E2E8EF", lw=.7, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#485A6D")

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(args.out / (name + "." + ext), dpi=300, facecolor="white",
                        bbox_inches="tight", pad_inches=.10, metadata={})
        plt.close(fig)

    fig = plt.figure(figsize=(12, 5.8))
    left = fig.add_axes([.08, .28, .16, .67])
    ax = fig.add_axes([.29, .28, .665, .67], sharey=left)
    for a in (left, ax):
        style(a); a.set_xscale("log"); a.set_ylim(13.8, 25.4)
        a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_locator(NullLocator())
    left.set_xlim(.015, .08); left.set_xticks([.02, .04, .08])
    ax.set_xlim(7, 1800); ax.set_xticks([10, 30, 100, 300, 1000])
    ax.tick_params(left=False, labelleft=False); ax.spines["left"].set_visible(False)
    left.set_ylabel("Shared-catalog PSNR (dB)", fontsize=15)
    for r in rows:
        if r["group"] != "gs" or r["condition"] != "d0": continue
        m = r["method"]; target = left if m == "random" else ax
        target.scatter(float(r["planning_wall_time_s"]), float(r["psnr_shared"]),
            marker="o" if r["scene"] == "interior_0007" else "^", s=130,
            facecolor="white" if m == "gavis" else colors[m], edgecolor=colors[m], lw=1.8, zorder=3)
    fig.text(.52, .17, "Adapter planning wall time (s; logarithmic axes with a gap)", ha="center", fontsize=14)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=colors[m],
        markerfacecolor="white" if m=="gavis" else colors[m], label=names[m]+(" (270 s)" if m=="gavis" else ""))
        for m in methods], loc="lower center", bbox_to_anchor=(.51,.065), ncol=6, frameon=False, fontsize=12)
    fig.text(.52, .012, "○ interior_0007     △ interior_0044     One acquisition seed; motion-only budget", ha="center", fontsize=12)
    save(fig, "figure-04-quality-and-planning-cost")

    fig = plt.figure(figsize=(12.5, 6.4)); ax = fig.add_axes([.08, .32, .89, .64]); style(ax)
    ax.axhline(0,color="#677787",lw=1)
    x=np.arange(6)
    for label, color, offset in [("severe",severe,-.18),("clean",clean,.18)]:
        deltas=np.array([[float(p["delta_"+label]) for p in pairs if p["group"]=="gs" and p["method"]==m] for m in methods])
        ax.bar(x+offset,deltas.mean(1),width=.33,color=color,alpha=.68)
        for k,marker in enumerate(("o","^")):
            ax.scatter(x+offset,deltas[:,k],marker=marker,s=45,facecolor="white",edgecolor=ink,zorder=4)
    ax.set_xticks(x,[names[m]+(" *" if m in ("gleam","gavis") else "") for m in methods])
    ax.set_ylabel("Dynamic − static PSNR (dB)",fontsize=15)
    fig.legend(handles=[Patch(color=severe,alpha=.68,label="Mean Δsevere"),Patch(color=clean,alpha=.68,label="Mean Δclean"),
                        Line2D([],[],marker="o",ls="",markerfacecolor="white",color=ink,label="interior_0007"),
                        Line2D([],[],marker="^",ls="",markerfacecolor="white",color=ink,label="interior_0044")],
               loc="lower center",bbox_to_anchor=(.52,.17),ncol=4,frameon=False,fontsize=12)
    fig.text(.08,.14,"Static baseline means (severe / clean, dB):",fontsize=12,color="#485A6D")
    for i,m in enumerate(methods):
        rr=[r for r in rows if r["group"]=="gs" and r["condition"]=="d0" and r["method"]==m]
        v=[np.mean([float(r["psnr_"+k]) for r in rr]) for k in ("severe","clean")]
        fig.text(.08+.89*(i+.5)/6,.09,f"{v[0]:.2f} / {v[1]:.2f}",ha="center",fontsize=12)
    fig.text(.08,.01,"* GAVIS: 270 s reference; dynamic GLEAM: 244.8 / 255.6 s. Markers show scenes, not seed uncertainty.",fontsize=11.5,color="#485A6D")
    save(fig,"figure-07-regional-dynamic-response")

    fig=plt.figure(figsize=(13,9.7)); ms=["r3con-pano","magician","fisherrf","gavis","random"]
    scenes=["apartment_1","mp3d_17DRP5sb8fy","skokloster_castle","van_gogh_room"]
    for idx,scene in enumerate(scenes):
        ri,ci=divmod(idx,2); left=.105+ci*.50; bottom=.56-ri*.45
        ax=fig.add_axes([left,bottom,.285,.31]);style(ax)
        ax.set_xlim(-20,6);ax.set_ylim(4.65,-.65)
        ax.set_yticks(range(5),[names[m] for m in ms],fontsize=12)
        ax.set_xticks([-20,-15,-10,-5,0,5]);ax.tick_params(axis="x",labelsize=11)
        ax.axvline(0,color="#657688",lw=1.1);ax.set_xlabel("Δ PSNR (dB)",fontsize=13)
        fig.text(left-.073,bottom+.354,scene,fontsize=15,weight="bold",color=ink)
        ax.text(1.06,1.04,"Static PSNR\nsevere / clean",transform=ax.transAxes,fontsize=11,color="#485A6D",va="bottom")
        for y,m in enumerate(ms):
            match=next((p for p in pairs if p["group"]=="mesh" and p["scene"]==scene and p["method"]==m),None)
            label="—"
            if match:
                ds,dc=float(match["delta_severe"]),float(match["delta_clean"])
                ax.plot([ds,dc],[y,y],color="#BAC3CC",lw=2,zorder=2)
                ax.scatter([ds,dc],[y,y],c=[severe,clean],s=65,zorder=3)
                label=f"{float(match['static_severe']):.2f} / {float(match['static_clean']):.2f}"
            else: ax.text(-8,y,"No matched pair",fontsize=10,ha="center",color="#7B8794")
            ax.text(1.06,y,label,transform=ax.get_yaxis_transform(),fontsize=11,va="center",color="#485A6D")
    fig.legend(handles=[Line2D([],[],marker="o",ls="",color=c,label=n) for c,n in [(severe,"Severe"),(clean,"Clean")]],
               loc="lower center",bbox_to_anchor=(.5,.006),ncol=2,frameon=False)
    save(fig,"figure-08-mesh-regional-dynamic-response")


if __name__ == "__main__":
    main()
