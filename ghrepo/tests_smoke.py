#!/usr/bin/env python3
"""Fast self-test (< 30 s): fault-injection operators behave as specified and a short Digits run completes.
Run:  python tests_smoke.py"""
import numpy as np, importlib.util, os, sys
H = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("dx", f"{H}/digits_experiment.py"); dx = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]; spec.loader.exec_module(dx)
rng = np.random.default_rng(0); t = rng.standard_normal((1, 64))
assert np.isnan(dx.inject(t, "nan", 1.0, 0, rng)).sum() == 1, "NaN model must corrupt exactly one element"
assert (dx.inject(t, "single", 10.0, 0, rng) != t).sum() == 1, "single model must corrupt exactly one element"
assert (dx.inject(t, "row", 10.0, 0, rng) != t).sum() == 64, "row model must scale the whole vector"
assert 5 <= (dx.inject(t, "block", 10.0, 0, rng) != t).sum() <= 7, "block model must corrupt ~10% contiguous features"
assert np.isfinite(dx.nanfilter(np.array([[np.nan, np.inf, 1.0]]))).all(), "NaN filter must remove non-finite values"
assert dx.relu6(np.array([-1.0, 3.0, 99.0])).tolist() == [0.0, 3.0, 6.0], "ReLU6 must clip to [0, 6]"
net = dx.Net([64, 32, 10], True, True, True, rng); net._rng = rng
for _ in range(20): net.train_step(dx.X_tr[:64], dx.Y_tr[:64], 0.05, True, 5.0)
acc = (net.predict(dx.X_te) == dx.y_te).mean(); assert acc > 0.5, f"short training should exceed chance, got {acc}"
print(f"smoke test passed (short-run accuracy {acc*100:.1f}%)")
