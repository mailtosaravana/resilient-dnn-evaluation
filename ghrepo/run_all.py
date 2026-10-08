#!/usr/bin/env python3
"""One-command reproduction of every result in the manuscript.
   python run_all.py --data /path/to/CIFAR-10-images   (≈ 70 min on 2 CPU cores)"""
import subprocess, sys, argparse, os
H = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(); p.add_argument("--data", default="/home/claude/data/CIFAR-10-images"); p.add_argument("--epochs", type=int, default=15); p.add_argument("--skip-cifar", action="store_true"); a = p.parse_args()
steps = [[sys.executable, f"{H}/digits_experiment.py"]]
if not a.skip_cifar: steps.append([sys.executable, f"{H}/cifar_experiment.py", "--epochs", str(a.epochs), "--data", a.data])
steps += [[sys.executable, f"{H}/make_diagrams.py"], [sys.executable, f"{H}/make_cifar_figures.py"], [sys.executable, f"{H}/make_tables.py"], [sys.executable, f"{H}/verify_results.py"]]
for s in steps:
    print("\n>>>", " ".join(s), flush=True); subprocess.run(s, check=True)
