#!/usr/bin/env python3
"""
ResilientNet reproducibility package -- Experiment 2 (convolutional network, CIFAR-10)

A compact residual network (ResNet-8: 3x3 stem + 3 residual stages of 16/32/64 channels) is trained on
CIFAR-10 in five configurations and evaluated with the same seven fault models as Experiment 1.
Faults are injected into the OUTPUT OF A CONVOLUTION (the matrix-multiplication result, before batch
normalisation / activation), at one of four sites: the stem convolution and the second convolution of
each residual stage. All outputs are produced from the same checkpoints.

  configurations : Baseline (Conv-BN-ReLU) | +ReLU6 | +ReLU6+OrderInv (Conv-ReLU6-BN) |
                   +ReLU6+OI+NaN (same weights as previous + inference NaN filter) | ResilientNet (FT)
  fault campaign : 7 fault models x 20 batches x 100 test images = 2,000 image-level trials per cell
  layer campaign : 4 sites x 10 batches x 100 images = 1,000 trials per site
  latency        : 30 timed forward passes of a 500-image batch per configuration (median, IQR)

Dataset: CIFAR-10 (Krizhevsky, 2009) read from the image-folder mirror cloned from
https://github.com/YoongiKim/CIFAR-10-images (identical 50,000 / 10,000 split).

Usage: python cifar_experiment.py [--epochs 15] [--data /path/to/CIFAR-10-images]
"""
import os, sys, json, time, argparse, glob
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import layers

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--data", default="/home/claude/data/CIFAR-10-images")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_cifar"))
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
tf.random.set_seed(args.seed); np.random.seed(args.seed)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FAULT_MODELS = ["single", "random", "row", "column", "block", "bitflip", "nan"]
MAGS = [1.5, 5.0, 10.0, 50.0, 100.0]

# ----------------------------------------------------------------------------- data
def load_split(split):
    cache = os.path.join(args.out, f"cifar_{split}.npz")
    if os.path.exists(cache):
        d = np.load(cache); return d["x"], d["y"]
    xs, ys = [], []
    for ci, c in enumerate(CLASSES):
        files = sorted(glob.glob(os.path.join(args.data, split, c, "*")))
        for f in files:
            xs.append(tf.io.decode_image(tf.io.read_file(f), channels=3).numpy()); ys.append(ci)
    x = np.stack(xs).astype(np.uint8); y = np.array(ys, dtype=np.int64)
    np.savez_compressed(cache, x=x, y=y); return x, y

print("loading CIFAR-10 ...", flush=True)
x_tr, y_tr = load_split("train"); x_te, y_te = load_split("test")
MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32); STD = np.array([0.2470, 0.2435, 0.2616], np.float32)
def prep(x): return ((x.astype(np.float32) / 255.0) - MEAN) / STD
X_tr, X_te = prep(x_tr), prep(x_te)
print(f"train {X_tr.shape}  test {X_te.shape}", flush=True)

# ----------------------------------------------------------------------------- fault injection (TF, per-sample random location)
@tf.function
def inject_tf(x, mode, mag, seed):
    """x: (N,H,W,C) float32. mode: int32 scalar 0..6. Each sample receives its own fault location."""
    N = tf.shape(x)[0]; H = tf.shape(x)[1]; W = tf.shape(x)[2]; C = tf.shape(x)[3]
    hh = tf.reshape(tf.range(H), (1, H, 1, 1)); ww = tf.reshape(tf.range(W), (1, 1, W, 1)); cc = tf.reshape(tf.range(C), (1, 1, 1, C))
    def rnd(maxv): return tf.reshape(tf.random.stateless_uniform((N,), seed=seed + tf.stack([0, maxv]), maxval=maxv, dtype=tf.int32), (N, 1, 1, 1))
    h0, w0, c0 = rnd(H), rnd(W), rnd(C)
    one_elem = (hh == h0) & (ww == w0) & (cc == c0)
    def single(): return tf.where(one_elem, x * mag, x)
    def random_(): return tf.where(tf.random.stateless_uniform(tf.shape(x), seed=seed + 7) < 0.01, x * mag, x)
    def row(): return tf.where((hh == h0) & (cc == c0), x * mag, x)
    def column(): return tf.where((ww == w0) & (cc == c0), x * mag, x)
    def block():
        bh = tf.maximum(1, H // 4); bw = tf.maximum(1, W // 4)
        hs = tf.minimum(h0, H - bh); ws = tf.minimum(w0, W - bw)
        m = (hh >= hs) & (hh < hs + bh) & (ww >= ws) & (ww < ws + bw) & (cc == c0)
        return tf.where(m, x * mag, x)
    def bitflip():
        bit = tf.random.stateless_uniform((N, 1, 1, 1), seed=seed + 11, maxval=32, dtype=tf.int32)
        flipped = tf.bitcast(tf.bitwise.bitwise_xor(tf.bitcast(x, tf.int32), tf.bitwise.left_shift(1, bit)), tf.float32)
        return tf.where(one_elem, flipped, x)
    def nan_(): return tf.where(one_elem, tf.fill(tf.shape(x), np.float32("nan")), x)
    return tf.switch_case(mode, [single, random_, row, column, block, bitflip, nan_])

# ----------------------------------------------------------------------------- model
class Block(tf.keras.layers.Layer):
    def __init__(self, ch, stride, relu6, oi, **kw):
        super().__init__(**kw)
        self.c1 = layers.Conv2D(ch, 3, stride, "same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.b1 = layers.BatchNormalization(); self.c2 = layers.Conv2D(ch, 3, 1, "same", use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.b2 = layers.BatchNormalization()
        self.sc = (layers.Conv2D(ch, 1, stride, "same", use_bias=False) if stride != 1 else None)
        self.sb = layers.BatchNormalization() if stride != 1 else None
        self.act = (lambda t: tf.nn.relu6(t)) if relu6 else (lambda t: tf.nn.relu(t)); self.oi = oi
    def cba(self, bn, t, training):   # conv output t -> normalised activation, in either order
        return bn(self.act(t), training=training) if self.oi else self.act(bn(t, training=training))
    def call(self, x, training=False, inj=None, site=None):
        s = x if self.sc is None else self.sb(self.sc(x), training=training)
        h = self.cba(self.b1, self.c1(x), training)
        h = self.c2(h)
        if inj is not None:          # fault in the second convolution output of this stage
            h = tf.cond(tf.equal(inj[0], site), lambda: inject_tf(h, inj[1], inj[2], inj[3]), lambda: h)
        h = self.cba(self.b2, h, training)
        return self.act(h + s)

class ResNet8(tf.keras.Model):
    def __init__(self, relu6=False, oi=False, nanfilter=False, **kw):
        super().__init__(**kw)
        self.stem = layers.Conv2D(16, 3, 1, "same", use_bias=False); self.sbn = layers.BatchNormalization()
        self.blocks = [Block(16, 1, relu6, oi), Block(32, 2, relu6, oi), Block(64, 2, relu6, oi)]
        self.act = (lambda t: tf.nn.relu6(t)) if relu6 else (lambda t: tf.nn.relu(t)); self.oi = oi; self.nf = nanfilter
        self.pool = layers.GlobalAveragePooling2D(); self.fc = layers.Dense(10)
    def filt(self, t): return tf.where(tf.math.is_finite(t), t, tf.zeros_like(t)) if self.nf else t
    def call(self, x, training=False, inj=None):
        h = self.stem(x)
        if inj is not None: h = tf.cond(tf.equal(inj[0], 0), lambda: inject_tf(h, inj[1], inj[2], inj[3]), lambda: h)
        h = self.sbn(self.act(h), training=training) if self.oi else self.act(self.sbn(h, training=training))
        h = self.filt(h)
        for i, b in enumerate(self.blocks):
            h = self.filt(b(h, training=training, inj=inj, site=i + 1))
        return self.fc(self.pool(h))

CONFIGS = [  # name, relu6, oi, nanfilter, fault_aware_training, weights_from
    ("Baseline (Conv-BN-ReLU)", False, False, False, False, None),
    ("+ReLU6",                   True,  False, False, False, None),
    ("+ReLU6+OrderInv",          True,  True,  False, False, None),
    ("+ReLU6+OI+NaN",            True,  True,  True,  False, "+ReLU6+OrderInv"),
    ("ResilientNet (FT)",        True,  True,  True,  True,  None),
]
BATCH = 128
steps_per_epoch = len(X_tr) // BATCH
sched = tf.keras.optimizers.schedules.CosineDecay(0.1, args.epochs * steps_per_epoch)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def augment(x):
    x = tf.image.random_flip_left_right(x)
    x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], "REFLECT")
    return tf.image.random_crop(x, tf.shape(x) - tf.constant([0, 8, 8, 0]))

def train(model, fault_aware):
    opt = tf.keras.optimizers.SGD(sched, momentum=0.9, nesterov=True)
    @tf.function
    def step(x, y, inj_on, inj_site, inj_mode, inj_mag, seed):
        with tf.GradientTape() as tape:
            inj = (inj_site, inj_mode, inj_mag, seed) if inj_on else None
            logits = model(augment(x), training=True, inj=inj)
            loss = loss_fn(y, logits) + tf.add_n(model.losses)
        g = tape.gradient(loss, model.trainable_variables); opt.apply_gradients(zip(g, model.trainable_variables))
        return loss
    # two traced variants: with and without injection (python bool -> separate traces)
    hist = []
    rng = np.random.default_rng(args.seed)
    for ep in range(args.epochs):
        t0 = time.time(); idx = rng.permutation(len(X_tr)); tot = 0.0
        fmag = np.float32(0.5 + (50.0 - 0.5) * ep / args.epochs)       # curriculum escalation
        for s in range(steps_per_epoch):
            b = idx[s * BATCH:(s + 1) * BATCH]
            on = bool(fault_aware and rng.random() < 0.75)
            site = np.int32(rng.integers(0, 4)); mode = np.int32(rng.integers(0, 5))   # five magnitude fault models
            seed = np.array([rng.integers(1 << 30), rng.integers(1 << 30)], np.int32)
            tot += float(step(X_tr[b], y_tr[b], on, site, mode, fmag, seed))
        acc = evaluate(model)
        hist.append({"epoch": ep + 1, "loss": tot / steps_per_epoch, "test_acc": acc, "time_s": time.time() - t0})
        print(f"   ep {ep+1:2d} loss {tot/steps_per_epoch:.3f} acc {acc*100:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return hist

@tf.function
def fwd(model, x): return model(x, training=False)
def evaluate(model):
    preds = np.concatenate([tf.argmax(fwd(model, X_te[i:i + 500]), 1).numpy() for i in range(0, len(X_te), 500)])
    return float((preds == y_te).mean())

@tf.function
def fwd_inj(model, x, site, mode, mag, seed): return model(x, training=False, inj=(site, mode, mag, seed))

def fi_campaign(model, name):
    """Returns application-level and site-level critical/SDC rates. One trial = one test image with one fault."""
    rng = np.random.default_rng(1000 + args.seed)
    clean_logits = np.concatenate([fwd(model, X_te[i:i + 500]).numpy() for i in range(0, len(X_te), 500)])
    clean_p = tf.nn.softmax(clean_logits).numpy(); clean_pred = clean_p.argmax(1)
    def run(mode_i, site_list, n_batches, mags):
        sdc = crit = n = 0
        for _ in range(n_batches):
            idx = rng.choice(len(X_te), 100, replace=False)
            site = np.int32(rng.choice(site_list)); mag = np.float32(rng.choice(mags)) if mags else np.float32(1.0)
            seed = np.array([rng.integers(1 << 30), rng.integers(1 << 30)], np.int32)
            out = fwd_inj(model, X_te[idx], site, np.int32(mode_i), mag, seed).numpy()
            finite = np.all(np.isfinite(out), 1)
            pf = np.where(finite[:, None], tf.nn.softmax(np.nan_to_num(out)).numpy(), 0.0)
            changed = ~np.all(np.isclose(pf, clean_p[idx], atol=1e-6), 1) | ~finite
            wrong = (pf.argmax(1) != clean_pred[idx]) | ~finite
            sdc += int(changed.sum()); crit += int(wrong.sum()); n += len(idx)
        return {"trials": n, "sdc_rate": sdc / n, "critical_rate": crit / n}
    app = {}
    for mi, fm in enumerate(FAULT_MODELS):
        app[fm] = run(mi, [0, 1, 2, 3], 20, MAGS if fm not in ("bitflip", "nan") else None)
        print(f"   {name:24s} {fm:8s} sdc {app[fm]['sdc_rate']*100:5.1f}  crit {app[fm]['critical_rate']*100:5.1f}", flush=True)
    site = {}
    for s in range(4):
        r = {"trials": 0, "sdc": 0, "crit": 0}
        for mi in range(5):
            o = run(mi, [s], 2, MAGS); r["trials"] += o["trials"]; r["sdc"] += o["sdc_rate"] * o["trials"]; r["crit"] += o["critical_rate"] * o["trials"]
        site[f"site{s}"] = {"trials": r["trials"], "sdc_rate": r["sdc"] / r["trials"], "critical_rate": r["crit"] / r["trials"]}
    return app, site

def latency(model):
    xb = X_te[:500]; fwd(model, xb)
    t = []
    for _ in range(30):
        t0 = time.perf_counter(); fwd(model, xb); t.append((time.perf_counter() - t0) * 1e3)
    t = np.array(t); return {"median_ms": float(np.median(t)), "iqr_ms": [float(np.percentile(t, 25)), float(np.percentile(t, 75))], "batch": 500}

# ----------------------------------------------------------------------------- main
results = {"seed": args.seed, "epochs": args.epochs, "configs": {}}
models = {}
for name, r6, oi, nf, ft, wfrom in CONFIGS:
    print(f"\n=== {name} ===", flush=True)
    tf.random.set_seed(args.seed)
    m = ResNet8(r6, oi, nf); m(X_te[:2])
    if wfrom is None:
        hist = train(m, ft); m.save_weights(os.path.join(args.out, f"weights_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'p').replace('-', '_')}.weights.h5"))
    else:
        m.set_weights(models[wfrom].get_weights()); hist = results["configs"][wfrom]["history"]
    models[name] = m
    acc = evaluate(m)
    app, site = fi_campaign(m, name)
    results["configs"][name] = {"clean_accuracy": acc, "history": hist, "fi_application": app, "fi_site": site,
                                "params": int(m.count_params()),
                                "avg_critical_rate_7models": float(np.mean([app[f]["critical_rate"] for f in FAULT_MODELS])),
                                "avg_sdc_rate_7models": float(np.mean([app[f]["sdc_rate"] for f in FAULT_MODELS]))}
    print(f"   acc {acc*100:.2f}  avgCrit {results['configs'][name]['avg_critical_rate_7models']*100:.1f}", flush=True)
    with open(os.path.join(args.out, "results_cifar.json"), "w") as f: json.dump(results, f, indent=1)
# matched latency: round-robin over all configurations so that drift affects every model equally
print("\nlatency (round-robin, 500-image batch, 40 rounds) ...", flush=True)
xb = X_te[:500]
for n in models: [fwd(models[n], xb) for _ in range(5)]
times = {n: [] for n in models}
for _ in range(40):
    for n in models:
        t0 = time.perf_counter(); fwd(models[n], xb); times[n].append((time.perf_counter() - t0) * 1e3)
for n in models:
    t = np.array(times[n]); results["configs"][n]["latency"] = {"median_ms": float(np.median(t)), "iqr_ms": [float(np.percentile(t, 25)), float(np.percentile(t, 75))], "batch": 500}
    print(f"   {n:24s} {np.median(t):7.1f} ms  IQR [{np.percentile(t,25):.1f}, {np.percentile(t,75):.1f}]", flush=True)
base = results["configs"]["Baseline (Conv-BN-ReLU)"]["avg_critical_rate_7models"]
for n in results["configs"]:
    results["configs"][n]["improvement_vs_baseline"] = base / max(results["configs"][n]["avg_critical_rate_7models"], 1e-9)
results["total_fault_injection_trials"] = sum(sum(v["trials"] for v in c["fi_application"].values()) + sum(v["trials"] for v in c["fi_site"].values()) for c in results["configs"].values())
with open(os.path.join(args.out, "results_cifar.json"), "w") as f: json.dump(results, f, indent=1)
print("\nTOTAL CIFAR fault-injection trials:", results["total_fault_injection_trials"])
