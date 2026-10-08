"""Standard-library Slurm launcher; every setting lives in config.py."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseline.UAVRTV.common.settings import load, validate, DEFAULT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "smoke", "train", "eval"), default="train")
    p.add_argument("--settings", type=Path, default=DEFAULT)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    s = validate(load(a.settings))
    command = ["sbatch", f"--chdir={s.PROJECT_ROOT}", f"--partition={s.PARTITION}",
        f"--cpus-per-task={s.CPUS}", f"--mem={s.MEMORY}", f"--time={s.TIME_LIMIT}",
        f"--job-name={s.JOB_NAME}-{a.mode}", f"--output={s.LOG_DIR}/%x-%j.out", f"--error={s.LOG_DIR}/%x-%j.err"]
    if s.GPUS:
        command.append(f"--gres=gpu:{s.GPUS}")
    command += [str(Path(__file__).with_name("job.sbatch")), str(s.PYTHON), a.mode, str(s.SETTINGS_PATH)]
    print(shlex.join(command), flush=True)
    print(f"Mode={a.mode}; train output={s.OUT}; eval output={s.EVAL_OUT}", flush=True)
    if a.dry_run:
        return 0
    if not Path(s.PYTHON).is_file():
        raise ValueError("Configured lab Python missing: " + str(s.PYTHON))
    s.LOG_DIR.mkdir(parents=True, exist_ok=True)
    frozen = s.LOG_DIR / (s.JOB_NAME + "-" + a.mode + "-settings-" + uuid.uuid4().hex + ".py")
    frozen.write_text("from pathlib import Path, PosixPath\n" + "\n".join(
        key + " = " + repr(value) for key, value in vars(s).items()
        if key.isupper() and key != "SETTINGS_PATH") + "\n")
    command[-1] = str(frozen.resolve())
    return subprocess.run(command, cwd=s.PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
