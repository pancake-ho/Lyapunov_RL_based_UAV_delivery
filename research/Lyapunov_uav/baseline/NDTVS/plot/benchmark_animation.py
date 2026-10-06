"""Shared RSU/UAV renderer from audited actual slot records; RSU-only has no UAV."""
import json
from pathlib import Path
import numpy as np
from baseline.NDTVS.common.config import read_config
from baseline.NDTVS.evaluation.benchmark.runner import verify_result
from baseline.NDTVS.rewards.qoe import PSNR_DB


def slot_figure(record,cfg,name,algorithm):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes = plt.subplots(3,1,figsize=(15,8),gridspec_kw=dict(height_ratios=(1.8,1,1)),layout="constrained")
    ax,qax,dax = axes
    users = sorted([u for r in record["regions"].values() for u in r["users"]],key=lambda u:u["user"])
    colors = {0:"#8795a1",1:"#2166b1",2:"#d17a14"}
    for m in range(cfg.num_regions):
        left,right = m*cfg.region_length_m,(m+1)*cfg.region_length_m
        ax.axvspan(left,right,color="#edf2f5" if m%2 else "#f9fafb")
        ax.axvline(left,color="#b8c2ca",ls=":",lw=.8)
        ax.scatter(cfg.rsu_x(m),0,marker="^",s=100,color=colors[1],zorder=5)
        ax.text(cfg.rsu_x(m),-.4,f"RSU {m}",ha="center",fontsize=8)
        rg = record["regions"][str(m)]
        if algorithm == "proposed" and rg["hired"]:
            ax.scatter(rg["uav_x"],1.1,marker="X",s=100,color=colors[2],zorder=5)
            ax.text(rg["uav_x"],1.5,f"UAV {m}",ha="center",fontsize=8)
        for u in rg["users"]:
            x,p = u["x_m"],u["provider"]
            ax.scatter(x,0,marker="o",s=28,color=colors[p],zorder=6)
            ax.annotate(f"u{u['user']}",(x,0),xytext=(0,10+(u["user"]%3)*11),textcoords="offset points",ha="center",fontsize=6)
            if p:
                start = cfg.rsu_x(m) if p == 1 else rg["uav_x"]
                ax.annotate("",(x,0),(start,0 if p == 1 else 1.1),arrowprops=dict(arrowstyle="->",color=colors[p],alpha=.5,lw=.8,connectionstyle="arc3,rad=.2"))
    ax.set(xlabel="Ground x (m); displayed vertical offsets are labels",yticks=[],ylim=(-.6,2))
    ax.legend(handles=[Line2D([],[],marker="o",ls="",color=c,label=n) for p,c,n in
        [(0,colors[0],"Unscheduled"),(1,colors[1],"RSU scheduled"),(2,colors[2],"UAV scheduled")] if p != 2 or algorithm == "proposed"],loc="upper left",ncol=3,fontsize=8)
    ids = [u["user"] for u in users]
    qax.bar(ids,[u["q_after"] for u in users],color=["#b2182b" if u["stall"] else colors[u["provider"]] for u in users])
    qax.set(ylabel="Buffer after slot (chunks)",ylim=(0,cfg.large_queue_level*1.05),xticks=ids)
    qax.tick_params(axis="x",labelsize=6)
    qax.set_title("Red: playback stalled during this slot",fontsize=9)
    dax.bar(np.array(ids)-.2,[u["req_chunks"] for u in users],width=.4,label="Requested",color="#b8c7d7")
    quality_colors = ("#fee08b","#a6d96a","#66c2a5","#3288bd")
    dax.bar(np.array(ids)+.2,[u["delivered"] for u in users],width=.4,label="Received",color=[quality_colors[u["req_quality"]] for u in users])
    dax.set(xlabel="Every user, including unserved users",ylabel="Chunks this slot",xticks=ids,ylim=(0,cfg.max_chunks_per_slot+.5))
    dax.tick_params(axis="x",labelsize=6)
    dax.legend(fontsize=8)
    dax.set_title("Received-bar colors follow the 34.00, 36.64, 39.11, 41.64 dB PSNR ladder",fontsize=9)
    fig.suptitle(f"{name} | episode {record['episode']} | frame {record['frame']} | slot {record['slot_in_frame']}\n"
                 "Actual slot-start positions and scheduling; buffers after slot; RSU-only methods display no UAV")
    return fig


def export(s,mode="sweep"):
    import baseline.NDTVS.api as c
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    state = verify_result(s.OUT/mode,c)
    for cell in state["cells"].values():
        if cell["experiment"] != "snr" or cell["seed"] not in s.VISUAL_SEEDS:
            continue
        if s.VISUAL_SNR_DB is not None and cell["snr_db"] not in s.VISUAL_SNR_DB:
            continue
        row = cell["rows"][0]  # Audited representative; all scenarios remain in CSVs.
        directory = s.OUT/mode/row["episode_dir"]
        cfg = read_config(directory/"resolved_config.json")
        records = [json.loads(line) for line in (directory/"trace.jsonl").read_text().splitlines()]
        slots = [r for r in records if r["event"] == "slot"]
        slots = slots[::s.VISUAL_SLOT_STRIDE][:s.VISUAL_MAX_IMAGES]
        output = s.OUT/mode/"visuals"/f"{cell['model']}_snr{cell['snr_db']}_seed{cell['seed']}"
        output.mkdir(parents=True,exist_ok=True)
        frames = []
        for record in slots:
            fig = slot_figure(record,cfg,cell["model"],state["spec"]["selection"][cell["model"]]["algorithm"])
            path = output/f"frame{record['frame']:04d}_slot{record['slot_in_frame']:03d}.png"
            fig.savefig(path,dpi=110)
            plt.close(fig)
            with Image.open(path) as image:
                image.thumbnail((1200,640))
                frames.append(image.convert("RGB"))
        if frames:
            frames[0].save(output/"animation.gif",save_all=True,append_images=frames[1:],duration=max(1,int(1000/s.GIF_FPS)),loop=0)
            for image in frames:
                image.close()
        print(f"Images / animation: {output}",flush=True)
    return 0
