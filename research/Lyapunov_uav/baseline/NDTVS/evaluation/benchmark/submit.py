"""Standard-library Slurm launcher. All resource/experiment knobs are in config.py."""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import argparse
import json
import shlex
import subprocess
from baseline.NDTVS.evaluation.benchmark.settings import load,validate,DEFAULT_CONFIG


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings",type=Path,default=DEFAULT_CONFIG)
    p.add_argument("--mode",choices=("all","smoke","sweep"),default="all")
    p.add_argument("--dry-run",action="store_true")
    a = p.parse_args(argv)
    s = validate(load(a.settings))
    job = Path(__file__).with_name("job.sbatch")
    command = ["sbatch",f"--chdir={s.PROJECT_ROOT}",f"--partition={s.PARTITION}",
        f"--cpus-per-task={s.CPUS}",f"--mem={s.MEMORY}",f"--time={s.TIME_LIMIT}",
        f"--job-name={s.JOB_NAME}",f"--output={s.LOG_DIR}/%x-%j.out",f"--error={s.LOG_DIR}/%x-%j.err"]
    if s.GPUS:
        command.append(f"--gres=gpu:{s.GPUS}")
    command += [str(job),str(s.PYTHON),a.mode,str(s.SETTINGS_PATH)]
    print(shlex.join(command))
    print(f"Output: {s.OUT}; seeds={s.SEEDS}; checkpoints={[m['name'] for m in s.MODELS]}")
    if a.dry_run:
        return 0
    if not Path(s.PYTHON).is_file():
        raise ValueError(f"Configured lab Python missing: {s.PYTHON}")
    s.LOG_DIR.mkdir(parents=True,exist_ok=True)
    # Each job receives an immutable copy of settings, isolating it from later edits.
    import uuid
    frozen = s.LOG_DIR/(s.JOB_NAME+"-settings-"+uuid.uuid4().hex+".py")
    # Freeze evaluated values, not __file__-relative expressions. No torch import.
    frozen.write_text("from pathlib import Path, PosixPath\n"+"\n".join(
        key+" = "+repr(value) for key,value in vars(s).items()
        if key.isupper() and key != "SETTINGS_PATH")+"\n")
    command[-1] = str(frozen.resolve())
    return subprocess.run(command,check=False,cwd=s.PROJECT_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
