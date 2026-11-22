"""Run from the project root with python experiment.py [--reference]."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
from time import perf_counter

# Set BLAS limits before importing NumPy. Small matrices do not need a thread pool.
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

from neuralforge.check import demo_check
from neuralforge.data import spirals
from neuralforge.network import MLP
from neuralforge.plots import figures
from neuralforge.training import fit


def paired_runs(options, original_test):
    """Keep the original seed triple as run one, then advance all three seeds."""
    runs = []
    for offset in range(5):
        data_seed, init_seed, batch_seed = 41 + offset, 17 + offset, 7 + offset
        train, validation, test = spirals(seed=data_seed)
        settings = options | {"seed": batch_seed}
        if offset == 0:
            mlp_test = original_test
        else:
            model = MLP([2, 32, 32, 2], seed=init_seed)
            fit(model, train, validation, **settings)
            mlp_test = model.metrics(*test)
        linear = MLP([2, 2], seed=init_seed)
        fit(linear, train, validation, **settings)
        runs.append({"data_seed": data_seed, "init_seed": init_seed, "batch_seed": batch_seed,
                     "mlp": mlp_test, "linear": linear.metrics(*test)})

    aggregate = {}
    for name in ("mlp", "linear"):
        aggregate[name] = {}
        for metric in ("accuracy", "loss"):
            values = [row[name][metric] for row in runs]
            aggregate[name][metric] = {"mean": float(np.mean(values)),
                                       "sample_sd": float(np.std(values, ddof=1))}
    return {"runs": runs, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true", help="Also train the matched CPU PyTorch model.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    start = perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    train, validation, test = spirals()
    options = dict(epochs=100, batch_size=64, method="adam", lr=0.01, seed=7)
    initial = MLP([2, 32, 32, 2])
    model = deepcopy(initial)
    tic = perf_counter()
    history = fit(model, train, validation, **options)
    report = {"python": platform.python_version(), "numpy": np.__version__, "settings": options,
              "architecture": [2, 32, 32, 2], "activation": "tanh", "data_seed": 41, "init_seed": 17,
              "numpy_fit_seconds": perf_counter() - tic, "gradient_check": demo_check(),
              "first_epoch_95_val": next((i for i, row in enumerate(history)
                                          if row["validation"]["accuracy"] >= 0.95), None)}
    experiments = []
    for width, lr, method in [(32, 0.01, "adam"), (32, 0.001, "adam"), (32, 0.03, "adam"),
                              (8, 0.01, "adam"), (64, 0.01, "adam"), (32, 0.01, "sgd")]:
        if (width, lr, method) == (32, 0.01, "adam"):
            last = history[-1]
        else:
            candidate = MLP([2, width, width, 2])
            last = fit(candidate, train, validation, **(options | {"lr": lr, "method": method}))[-1]
        experiments.append(dict(width=width, lr=lr, method=method, **last["validation"]))
    report["experiments"] = experiments
    report["initial_test"] = initial.metrics(*test)
    report["final_test"] = model.metrics(*test)
    report["robustness"] = paired_runs(options, report["final_test"])
    if args.reference:
        from neuralforge.reference import compare
        report["reference"] = compare(initial, model, train, test, **options)
    figures(out, initial, model, test, history)
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    report["pipeline_seconds"] = perf_counter() - start
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
