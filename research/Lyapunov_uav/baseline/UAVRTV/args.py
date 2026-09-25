"""
Command-line arguments for the SAC-based UAV video-delivery baseline
(Wu et al., IEEE IoT-J 2024) re-implemented inside the
RSU + UAV vehicular video-delivery scenario (frame/slot, playback queue,
quality ladder, physical battery, hiring cost).

Every physical / algorithmic constant is exposed here so the baseline can be
calibrated from the shell on Ubuntu without touching the code.
"""
import argparse
import json
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def float_list(s):
    return [float(x) for x in s.replace(",", " ").split()]


def build_parser():
    p = argparse.ArgumentParser(
        description="SAC (Wu et al. 2024) baseline for RSU+UAV vehicular video delivery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ------------------------------------------------------------------ run
    g = p.add_argument_group("run")
    g.add_argument("--agent", choices=["sac", "random"], default="sac",
                   help="sac = reference-paper agent; random = uniform random actions (env sanity check)")
    g.add_argument("--episodes", type=int, default=200)
    g.add_argument("--frames_per_episode", type=int, default=60, help="episode length in frames")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu")
    g.add_argument("--log_dir", default="runs/sac_baseline")
    g.add_argument("--save_path", default=None, help="checkpoint path (default: <log_dir>/sac.pt)")
    g.add_argument("--load_path", default=None, help="load checkpoint before running")
    g.add_argument("--eval_only", type=str2bool, default=False, help="deterministic policy, no training")
    g.add_argument("--eval_every", type=int, default=20, help="run one deterministic eval episode every N episodes")

    # -------------------------------------------------------------- logging
    g = p.add_argument_group("logging")
    g.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    g.add_argument("--slot_log", type=str2bool, default=True,
                   help="write per-slot state/decision/next-state debug log (text + jsonl)")
    g.add_argument("--slot_log_every_ep", type=int, default=1,
                   help="write the slot log only for episodes with ep %% N == 0 (eval episodes always logged)")
    g.add_argument("--console_slot_log", type=str2bool, default=False,
                   help="also print the per-slot text log to stdout")

    # ------------------------------------------------------------- scenario
    g = p.add_argument_group("scenario (road / RSU / traffic)")
    g.add_argument("--num_rsu", type=int, default=2, help="number of RSUs = regions = UAVs (one UAV per region)")
    g.add_argument("--rsu_spacing", type=float, default=400.0, help="RSU spacing [m]; region = road segment of this length")
    g.add_argument("--rsu_height", type=float, default=10.0)
    g.add_argument("--user_height", type=float, default=1.5)
    g.add_argument("--lane_y", type=float, default=3.5, help="lane offset [m]; +y lane drives +x, -y lane drives -x")
    g.add_argument("--flight_y_max", type=float, default=60.0, help="UAV allowed |y| [m]")
    g.add_argument("--flight_x_margin", type=float, default=100.0, help="UAV may leave its region by this margin [m]")
    g.add_argument("--init_vehicles_per_region", type=int, default=8)
    g.add_argument("--arrival_rate", type=float, default=0.15, help="Poisson vehicle arrivals per direction per second")
    g.add_argument("--veh_speed_min", type=float, default=5.0)
    g.add_argument("--veh_speed_max", type=float, default=20.0)
    g.add_argument("--hotspot_region", type=int, default=0, help="region index with congestion (-1 = none)")
    g.add_argument("--hotspot_speed_factor", type=float, default=0.3, help="vehicle speed multiplier inside hotspot region")
    g.add_argument("--max_users_per_region", type=int, default=12,
                   help="N_max: fixed observation/action slots per region; extra users are unserved")

    # ---------------------------------------------------------------- time
    g = p.add_argument_group("time structure")
    g.add_argument("--slot_duration", type=float, default=1.0, help="Delta [s]")
    g.add_argument("--slots_per_frame", type=int, default=10, help="T slots per frame")

    # --------------------------------------------------------------- radio
    g = p.add_argument_group("radio")
    g.add_argument("--W_R", type=float, default=20e6, help="RSU video bandwidth [Hz]")
    g.add_argument("--P_R", type=float, default=10.0, help="RSU total video power [W]")
    g.add_argument("--J_R", type=int, default=6, help="RSU max concurrent users")
    g.add_argument("--W_U", type=float, default=5e6, help="UAV video bandwidth [Hz]")
    g.add_argument("--P_U_max", type=float, default=2.0, help="UAV max total RF power [W]")
    g.add_argument("--J_U", type=int, default=3, help="UAV max concurrent users")
    g.add_argument("--fc", type=float, default=5e9, help="carrier frequency [Hz]")
    g.add_argument("--N0_dBm_Hz", type=float, default=-174.0)
    g.add_argument("--snr_gap_dB", type=float, default=3.0, help="Shannon gap Gamma [dB]")
    g.add_argument("--alpha_R", type=float, default=3.5, help="RSU path-loss exponent")
    g.add_argument("--alpha_U", type=float, default=2.2, help="UAV path-loss exponent (LoS-dominant)")
    g.add_argument("--beta_R_dB", type=float, default=None, help="RSU reference gain at 1 m [dB] (default: free space at fc)")
    g.add_argument("--beta_U_dB", type=float, default=None, help="UAV reference gain at 1 m [dB] (default: free space at fc)")
    g.add_argument("--fading", choices=["none", "rayleigh"], default="rayleigh",
                   help="small-scale fading xi (exp(1) per slot) applied to both links")
    g.add_argument("--bw_mode", choices=["sac", "fixed"], default="sac",
                   help="sac = SAC allocates UAV bandwidth/power among its users (reference paper); "
                        "fixed = fixed reserved resource block W_U/J_U, P_U_max/J_U (scenario default)")

    # --------------------------------------------------------------- video
    g = p.add_argument_group("video / playback")
    g.add_argument("--bitrates_mbps", type=float_list, default=[0.5, 1.0, 2.0, 4.0], help="quality ladder [Mbps]")
    g.add_argument("--utilities", type=float_list, default=[0.4, 0.6, 0.8, 1.0], help="normalized utility U_k")
    g.add_argument("--chunk_duration", type=float, default=1.0, help="playback seconds per chunk (S_k = R_k * this)")
    g.add_argument("--b", type=int, default=1, help="playback demand [chunks/slot]")
    g.add_argument("--L_max", type=int, default=4, help="max chunks per user per slot")
    g.add_argument("--Q_tilde", type=int, default=50, help="large-buffer constant Qe [chunks]; Z = Qe - Q")
    g.add_argument("--Q_init", type=int, default=5, help="prebuffered chunks for a new user")
    g.add_argument("--rsu_rule", choices=["maxq", "maxchunks"], default="maxq",
                   help="RSU per-slot rule (not part of the SAC agent): maxq = highest feasible quality, "
                        "maxchunks = quality that maximizes delivered chunks")

    # ------------------------------------------------------------ UAV energy
    g = p.add_argument_group("UAV propulsion / battery (rotary-wing model, Zeng et al.)")
    g.add_argument("--P0", type=float, default=79.86, help="blade profile power [W]")
    g.add_argument("--Pi", type=float, default=88.63, help="induced power [W]")
    g.add_argument("--U_tip", type=float, default=120.0, help="rotor tip speed [m/s]")
    g.add_argument("--v0", type=float, default=4.03, help="mean rotor induced velocity [m/s]")
    g.add_argument("--d0", type=float, default=0.6, help="fuselage drag ratio")
    g.add_argument("--rho", type=float, default=1.225, help="air density")
    g.add_argument("--rotor_solidity", type=float, default=0.05)
    g.add_argument("--rotor_area", type=float, default=0.503)
    g.add_argument("--uav_altitude", type=float, default=50.0)
    g.add_argument("--v_max", type=float, default=20.0, help="UAV max horizontal speed [m/s]")
    g.add_argument("--E_max_Wh", type=float, default=100.0, help="battery capacity [Wh]")
    g.add_argument("--E_th_frac", type=float, default=0.2, help="automatic-return reserve as fraction of E_max")
    g.add_argument("--P_ch", type=float, default=300.0, help="charging power at depot [W]")
    g.add_argument("--eta_ch", type=float, default=0.9, help="charging efficiency")
    g.add_argument("--eta_PA", type=float, default=0.35, help="power-amplifier efficiency for RF energy")
    g.add_argument("--e_rel", type=float, default=2000.0, help="fixed energy charged for a depot return [J]")

    # ------------------------------------------------------- utility / reward
    g = p.add_argument_group("reward (reference eq. 13-14 adapted)")
    g.add_argument("--beta", type=float, default=1.0, help="video-quality reward weight")
    g.add_argument("--delta", type=float, default=1e-6, help="bitrate-switch penalty [$/bps]")
    g.add_argument("--phi", type=float, default=10.0, help="stall (delay) penalty per user per slot")
    g.add_argument("--varsigma", type=float, default=1e-3, help="UAV energy penalty [$/J]")
    g.add_argument("--c_H", type=float, default=1.0, help="UAV hiring price per frame")
    g.add_argument("--lambda_H", type=float, default=1.0, help="hiring-cost weight")
    g.add_argument("--quality_reward", choices=["per_chunk", "per_user"], default="per_chunk",
                   help="per_chunk: beta*sum_n l_n U_k ; per_user: beta*sum_n U_k*1{l_n>0} (closer to one-GOP-per-slot)")
    g.add_argument("--reward_scope", choices=["region", "uav"], default="region",
                   help="QoE terms summed over all users in the region, or only UAV-served users")
    g.add_argument("--hire_mode", choices=["sac", "always", "never", "threshold"], default="sac",
                   help="who decides UAV hiring at frame start")

    # ----------------------------------------------------------------- SAC
    g = p.add_argument_group("SAC")
    g.add_argument("--hidden", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3, help="actor/critic learning rate (paper: 0.001)")
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--tau", type=float, default=0.005, help="target smoothing")
    g.add_argument("--alpha", type=float, default=0.2, help="initial entropy temperature")
    g.add_argument("--auto_alpha", type=str2bool, default=True)
    g.add_argument("--batch_size", type=int, default=256)
    g.add_argument("--buffer_size", type=int, default=200_000)
    g.add_argument("--start_steps", type=int, default=2000, help="random-action warm-up transitions")
    g.add_argument("--updates_per_step", type=int, default=1)
    return p


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    assert len(args.bitrates_mbps) == len(args.utilities), "--bitrates_mbps and --utilities must have equal length"
    assert args.J_U >= 1 and args.J_R >= 1
    assert args.max_users_per_region >= args.J_R + args.J_U, \
        "--max_users_per_region should be >= J_R + J_U so all schedulable users have observation slots"
    if args.save_path is None:
        args.save_path = os.path.join(args.log_dir, "sac.pt")
    return args


def save_args(args, path):
    with open(path, "w") as f:
        json.dump(vars(args), f, indent=2)
