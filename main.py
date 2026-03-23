import subprocess
import sys
from pathlib import Path


def run_script(script_path):
    print("\n" + "=" * 60)
    print(f"Running: {script_path}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"\nError while running {script_path}")
        sys.exit(result.returncode)

    print(f"Finished: {script_path}")


def main():
    project_root = Path(__file__).resolve().parent

    scripts = [
        project_root / "q1" / "mfcc_manual.py",
        project_root / "q1" / "leakage_snr.py",
        project_root / "q1" / "voiced_unvoiced.py",
        project_root / "q1" / "phonetic_mapping.py",
        project_root / "q2" / "train.py",
        project_root / "q2" / "eval.py",
        project_root / "q3" / "audit.py",
        project_root / "q3" / "pp_demo.py",
        project_root / "q3" / "train_fair.py",
        project_root / "q3" / "evaluation_scripts" / "proxy_metrics.py",
    ]

    for script in scripts:
        if not script.exists():
            print(f"Missing file: {script}")
            sys.exit(1)

    for script in scripts:
        run_script(str(script))

    print("\nAll questions executed successfully.")


if __name__ == "__main__":
    main()