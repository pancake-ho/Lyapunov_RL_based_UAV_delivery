"""Paired scenario intervals, coverage/fairness/CDFs and explicit cost analysis."""
import itertools
import json
from pathlib import Path
import numpy as np
from baseline.NDTVS.common.io import atomic, write_rows
from baseline.NDTVS.evaluation.benchmark.runner import verify_result
from baseline.NDTVS.evaluation.benchmark.radio import axis_label

METRICS = (
    "stall_time_ratio", "stall_user_slot_ratio", "average_quality_utility",
    "average_received_psnr_db", "unique_served_user_ratio", "never_served_user_ratio",
    "delivery_user_slot_ratio", "scheduled_user_slot_ratio", "playback_fulfillment_ratio",
    "delivery_jain_index", "playback_jain_index", "top20_delivery_share",
    "worst_user_stall_time_ratio", "p90_user_stall_time_ratio",
    "p10_user_delivery_chunks_per_slot", "per_user_quality_mean",
    "paper_qoe_per_user_slot", "paper_qoe_final_per_user", "request_failure_ratio",
    "delivered_chunks_per_user_slot", "hire_rate", "energy_consumed_j",
    "hired_uav_frames", "hiring_cost_total", "cost_augmented_qoe_per_user_slot",
    "cost_augmented_qoe_final_per_user")
QUALITY = {"average_quality_utility", "average_received_psnr_db"}


def estimates(rows,metric,samples):
    if metric in QUALITY:
        weights = np.array([r["received_segments_total"] for r in rows],float)
        values = np.array([r[metric] for r in rows],float)
        total = weights.sum()
        value = float(values@weights/total) if total else None
        denominators = weights[samples].sum(1)
        boot = np.divide((values*weights)[samples].sum(1),denominators,
                         out=np.full(len(samples),np.nan),where=denominators>0)
        valid = int(np.count_nonzero(weights))
    else:
        values = np.array([np.nan if r.get(metric) is None else r[metric] for r in rows],float)
        valid = int(np.isfinite(values).sum())
        value = float(np.mean(values[np.isfinite(values)])) if valid else None
        # Undefined all-zero-service fairness stays undefined; never treat it as 1.
        boot = values[samples].mean(1) if valid == len(rows) else np.full(len(samples),np.nan)
    ci = np.quantile(boot,[.025,.975]).tolist() if len(rows)>1 and np.isfinite(boot).all() else [None,None]
    return value,ci,boot,valid


def grouped_rows(state,experiment="snr"):
    groups = {}
    for cell in state["cells"].values():
        if cell["experiment"] != experiment:
            continue
        x = cell["snr_db"] if experiment == "snr" else cell["hiring_cost_per_frame"]
        group = groups.setdefault((x,cell["model"]),[])
        group.extend((cell["seed"],row["episode"],row) for row in cell["rows"])
    return {key:sorted(value,key=lambda v:(v[0],v[1])) for key,value in groups.items()}


def summarise(groups,s):
    if not groups:
        return [],[]
    first = next(iter(groups.values()))
    identities = [(seed,ep) for seed,ep,_ in first]
    if any([(seed,ep) for seed,ep,_ in rows] != identities for rows in groups.values()):
        raise ValueError("Unequal scenario pairing")
    # Stratified bootstrap: resample test episodes within each scenario-seed stratum.
    rng = np.random.default_rng(s.BOOTSTRAP_SEED)
    columns = []
    for seed in sorted({k[0] for k in identities}):
        indices = [i for i,pair in enumerate(identities) if pair[0] == seed]
        columns.append(rng.choice(indices,size=(s.BOOTSTRAP_SAMPLES,len(indices)),replace=True))
    samples = np.concatenate(columns,axis=1)
    summaries, differences, cached = [],[],{}
    for (x,name),records in groups.items():
        rows = [r for _,_,r in records]
        for metric in METRICS:
            if not any(metric in r for r in rows):
                continue
            estimate,ci,boot,valid = estimates(rows,metric,samples)
            if all(len(indices) == 1 for indices in ([i for i,p in enumerate(identities) if p[0] == seed] for seed in {k[0] for k in identities})):
                ci = [None,None]  # Smoke has no within-seed replication.
            summaries.append(dict(x=x,model=name,metric=metric,estimate=estimate,
                ci95_low=ci[0],ci95_high=ci[1],episodes=len(rows),defined_episodes=valid))
            cached[x,name,metric] = estimate,boot,ci
    for x in sorted({k[0] for k in groups}):
        for a,b in itertools.combinations(sorted({k[1] for k in groups if k[0] == x}),2):
            for metric in METRICS:
                if (x,a,metric) not in cached or (x,b,metric) not in cached:
                    continue
                av,ab,ac = cached[x,a,metric]
                bv,bb,bc = cached[x,b,metric]
                value = av-bv if av is not None and bv is not None else None
                ci = np.quantile(ab-bb,[.025,.975]).tolist() if ac[0] is not None and bc[0] is not None else [None,None]
                differences.append(dict(x=x,difference=f"{a} - {b}",metric=metric,estimate=value,ci95_low=ci[0],ci95_high=ci[1]))
    return summaries,differences


def panels(summaries,out,filename,panel_specs,xlabel,title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({r["model"] for r in summaries})
    fig,axes = plt.subplots(2,3,figsize=(15,8),squeeze=False)
    for ax,(metric,label,scale) in zip(axes.ravel(),panel_specs):
        for name in names:
            rows = sorted((r for r in summaries if r["model"] == name and r["metric"] == metric),key=lambda r:r["x"])
            if not rows:
                continue
            values = lambda key:np.array([np.nan if r[key] is None else r[key]*scale for r in rows])
            x = [r["x"] for r in rows]
            line, = ax.plot(x,values("estimate"),marker="o",label=name)
            ax.fill_between(x,values("ci95_low"),values("ci95_high"),alpha=.15,color=line.get_color())
        ax.set(xlabel=xlabel,ylabel=label)
        if metric in ("average_quality_utility","delivery_jain_index","playback_jain_index","top20_delivery_share","per_user_quality_mean"):
            ax.set_ylim(0,1.03)
        elif scale == 100:
            ax.set_ylim(0,103)
        ax.grid(alpha=.2)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=min(len(names),4),fontsize=9)
    fig.suptitle(title or "Mean over paired scenarios; shaded pointwise 95% scenario CI")
    fig.tight_layout(rect=(0,.06,1,.96))
    for ext in ("png","pdf"):
        fig.savefig(out/f"{filename}.{ext}",dpi=160)
    plt.close(fig)


def cdfs(state,root,out):
    import matplotlib.pyplot as plt
    from baseline.NDTVS.evaluation.checks import read_json
    levels = sorted(state["spec"]["snr_levels"])
    names = sorted(state["spec"]["selection"])
    for metric,label in (("stall_time_ratio","User stall time ratio"),("delivered_chunks_per_slot","Received chunks / slot")):
        fig,axes = plt.subplots(1,len(levels),figsize=(4*len(levels),4),squeeze=False)
        for ax,db in zip(axes[0],levels):
            for name in names:
                users = [u for cell in state["cells"].values() if cell["experiment"] == "snr" and cell["model"] == name and cell["snr_db"] == db
                         for row in cell["rows"] for u in read_json(root/row["per_user_file"])]
                x = np.sort([u[metric] for u in users])
                ax.step(x,np.arange(1,len(x)+1)/len(x),where="post",label=name)
            ax.set(xlabel=label,ylabel="Fraction of user-episode observations",title=f"SNR axis = {db} dB",ylim=(0,1))
            if metric == "stall_time_ratio":
                ax.set_xlim(0,1)
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle("Includes every user, including users with zero received chunks; descriptive CDF")
        fig.tight_layout()
        for ext in ("png","pdf"):
            fig.savefig(out/f"user_{metric}_cdf.{ext}",dpi=160)
        plt.close(fig)


def action_distributions(state,out):
    """Explain constant quality and one-vs-many-chunk behavior without thresholds."""
    import matplotlib.pyplot as plt
    groups = grouped_rows(state)
    levels = sorted(state["spec"]["snr_levels"])
    names = sorted(state["spec"]["selection"])
    fig,axes = plt.subplots(2,len(levels),figsize=(4*len(levels),8),squeeze=False)
    csv=[]
    for col,db in enumerate(levels):
        for name in names:
            rows = [r for _,_,r in groups[db,name]]
            for ax,kind,key in ((axes[0,col],"received_quality","received_quality_histogram"),
                               (axes[1,col],"requested_chunks","requested_chunk_histogram")):
                counts=np.sum([r[key] for r in rows],axis=0)
                probabilities=counts/counts.sum() if counts.sum() else np.full(len(counts),np.nan)
                ax.plot(range(len(counts)),probabilities,marker="o",label=name)
                csv.extend(dict(snr_db=db,model=name,kind=kind,index=i,count=int(count),
                    fraction=float(probabilities[i]) if counts.sum() else None) for i,count in enumerate(counts))
                ax.set(ylim=(0,1.03),ylabel="Fraction",title=f"SNR axis = {db} dB")
                ax.grid(alpha=.2)
        axes[0,col].set(xticks=range(4),xticklabels=["34.00","36.64","39.11","41.64"],xlabel="Received-chunk PSNR (dB)")
        axes[1,col].set(xlabel="Requested chunks per user-slot (0 includes unscheduled)")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Quality distribution is conditional on received chunks; chunk requests include every user-slot")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(out/f"quality_chunk_distributions.{ext}",dpi=160)
    plt.close(fig)
    write_rows(out/"quality_chunk_distributions.csv",csv)


def report(s,mode="sweep"):
    import baseline.NDTVS.api as c
    root,out = s.OUT/mode,s.OUT/mode/"summary"
    state = verify_result(root,c)
    out.mkdir(parents=True,exist_ok=True)
    groups = grouped_rows(state)
    summary,differences = summarise(groups,s)
    write_rows(out/"means_ci.csv",summary)
    write_rows(out/"paired_differences.csv",differences)
    per_seed = []
    episode_rows,all_users = [],[]
    for cell in state["cells"].values():
        for row in cell["rows"]:
            condition = {k:cell[k] for k in ("experiment","snr_db","seed","model","hiring_cost_per_frame")}
            episode_rows.append(dict(condition,**row))
            all_users.extend(dict(condition,episode=row["episode"],**u) for u in json.loads((root/row["per_user_file"]).read_text()))
        for metric in METRICS:
            if not any(metric in r for r in cell["rows"]):
                continue
            sample = np.array([list(range(len(cell["rows"])))])
            value,_,_,valid = estimates(cell["rows"],metric,sample)
            per_seed.append(dict(experiment=cell["experiment"],snr_db=cell["snr_db"],seed=cell["seed"],model=cell["model"],cost=cell["hiring_cost_per_frame"],metric=metric,estimate=value,defined_episodes=valid))
    write_rows(out/"per_seed.csv",per_seed)
    write_rows(out/"per_episode.csv",episode_rows)
    write_rows(out/"per_user.csv",all_users)
    spec = state["spec"]
    xlabel = axis_label(spec["snr_mode"],spec["reference_distance_m"])
    panels(summary,out,"performance",(("stall_time_ratio","Playback stall time (%)",100),
        ("average_quality_utility","Received-chunk weighted PSNR / 41.64",1),
        ("unique_served_user_ratio","Users receiving at least one chunk (%)",100),
        ("delivery_user_slot_ratio","User-slots with successful delivery (%)",100),
        ("playback_fulfillment_ratio","Playback demand satisfied (%)",100),
        ("stall_user_slot_ratio","User-slots with any stall (%)",100)),xlabel)
    panels(summary,out,"fairness",(("never_served_user_ratio","Users receiving zero chunks (%)",100),
        ("p90_user_stall_time_ratio","90th percentile user stall (%)",100),
        ("worst_user_stall_time_ratio","Worst-user stall (%)",100),
        ("delivery_jain_index","Jain index: received chunks, all users",1),
        ("top20_delivery_share","Top 20% users' share of chunks",1),
        ("p10_user_delivery_chunks_per_slot","10th percentile received chunks / slot",1)),xlabel)
    panels(summary,out,"qoe_resources",(("paper_qoe_final_per_user","Final NDTVS-adapted QoE / user",1),
        ("paper_qoe_per_user_slot","Mean cumulative QoE / user-slot",1),
        ("delivered_chunks_per_user_slot","Received chunks / user-slot",1),
        ("hire_rate","Hired UAV-region-frames (%)",100),
        ("energy_consumed_j","Episode UAV energy (J)",1),
        ("request_failure_ratio","Failures among requested user-slots (%)",100)),xlabel)
    cdfs(state,root,out)
    action_distributions(state,out)
    costgroups = grouped_rows(state,"cost")
    if spec["cost_mode"] == "accounting":
        for cost in spec["hiring_costs"]:
            for name in spec["selection"]:
                modified = []
                cfg = spec["configs"][name]
                n = cfg["num_regions"]*cfg["users_per_region"]
                slots = n*cfg["num_frames"]*cfg["frame_slots"]
                for seed,ep,row in groups[spec["cost_snr_db"],name]:
                    charge = spec["qoe_cost_weight"]*cfg["lambda_h"]*cost*row["hired_uav_frames"]
                    modified.append((seed,ep,dict(row,hiring_cost_total=cfg["lambda_h"]*cost*row["hired_uav_frames"],
                        cost_augmented_qoe_per_user_slot=row["paper_qoe_per_user_slot"]-charge/slots,
                        cost_augmented_qoe_final_per_user=row["paper_qoe_final_per_user"]-charge/n)))
                costgroups[cost,name] = modified
    costs,costdiff = summarise(costgroups,s)
    if costs:
        write_rows(out/"cost_means_ci.csv",costs)
        write_rows(out/"cost_paired_differences.csv",costdiff)
        panels(costs,out,"hiring_cost_sweep",(("cost_augmented_qoe_final_per_user","Cost-augmented final QoE / user",1),
            ("paper_qoe_final_per_user","Original final QoE / user",1),
            ("hire_rate","Hired UAV-region-frames (%)",100),
            ("stall_time_ratio","Playback stall time (%)",100),
            ("average_quality_utility","Received PSNR / 41.64",1),
            ("unique_served_user_ratio","Users receiving a chunk (%)",100)),"UAV hiring cost per region-frame",
            "Fixed trained networks; completion re-evaluated per cost" if spec["cost_mode"] == "reevaluate" else "Fixed actions; cost accounting only")
    from baseline.NDTVS.evaluation.checks import sha256
    atomic(out/"report.json",dict(spec=spec,report_source_sha256=sha256(Path(__file__)),means=summary,paired_differences=differences,cost_means=costs,
        quality_aggregation="Pooled received-segment weighted; no-reception quality is undefined.",
        fairness_aggregation="Per-episode all-user fairness then equal-horizon episode mean; defined counts disclosed.",
        bootstrap="Resample episodes within each scenario-seed stratum; same draws for each policy.",
        cost_definition="Cost-augmented final QoE = original final mean user QoE - w_cost*lambda_h*c_H*sum(hire)/N. Separate from paper QoE; no policy retraining.",
        initial_buffer="Included equally for all methods; playback can succeed initially without any delivery."))
    print(f"Graphs and CSVs: {out}",flush=True)
    return 0
