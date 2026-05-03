"""
Compare optimizer-identified parameters against the ground truth stored in
the simulated MCAPs from `tests/sim_collect.py`.

Reads:
  * any one MCAP from `--recordings` (their `recording` metadata must
    contain `true_armature`, `true_damping`, `true_frictionloss`; sim
    recordings produced together share the same truth)
  * `--out-dir/results.json` produced by `scripts/optimize.py`

Prints a table of true vs. identified vs. error and exits non-zero if any
relative error exceeds the configured tolerance.

Usage:
    python tests/check_recovery.py \
        --recordings data/sim/ \
        --out-dir results/sim/ \
        --rel-tol 0.30
"""



import argparse
import json
import sys
from pathlib import Path

# allow `from recording import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from recording import read_mcap  # noqa: E402


PARAM_NAMES = ("armature", "damping", "frictionloss")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings", type=Path, required=True,
        help="Directory of simulated *.mcap recordings (truth lives in metadata).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory containing results.json from optimize.py.",
    )
    parser.add_argument(
        "--rel-tol", type=float, default=0.30,
        help="Maximum allowed |identified-true|/|true|. Default 30%%.",
    )
    args = parser.parse_args()

    if not args.recordings.is_dir():
        raise SystemExit(f"--recordings must be a directory, got {args.recordings}")
    mcaps = sorted(args.recordings.glob("*.mcap"))
    if not mcaps:
        raise SystemExit(f"No *.mcap files in {args.recordings}")

    rec = read_mcap(mcaps[0])
    missing = [
        k for k in ("true_armature", "true_damping", "true_frictionloss")
        if k not in rec.metadata
    ]
    if missing:
        print(
            f"[!] {mcaps[0]} is missing ground-truth metadata {missing}. "
            "Run tests/sim_collect.py to produce simulated MCAPs.",
            file=sys.stderr,
        )
        sys.exit(2)

    truth = {
        "armature": float(rec.metadata["true_armature"]),
        "damping": float(rec.metadata["true_damping"]),
        "frictionloss": float(rec.metadata["true_frictionloss"]),
    }

    results_json = args.out_dir / "results.json"
    if not results_json.exists():
        raise SystemExit(
            f"{results_json} not found; run scripts/optimize.py first."
        )
    params = json.loads(results_json.read_text())
    init = params.get("initial", {})
    frozen = params.get("frozen", {})

    rows = []
    worst_rel = 0.0
    for name in PARAM_NAMES:
        true_v = truth[name]
        opt_v = float(params[name])
        init_v = float(init.get(name, float("nan")))
        is_frozen = bool(frozen.get(name, False))
        abs_err = opt_v - true_v
        rel_err = abs_err / true_v if true_v != 0 else float("inf")
        if not is_frozen:
            worst_rel = max(worst_rel, abs(rel_err))
        rows.append((name, true_v, init_v, opt_v, abs_err, rel_err, is_frozen))

    header = f"{'param':>14s} {'true':>11s} {'initial':>11s} {'identified':>13s} {'abs err':>11s} {'rel err':>9s}"
    print(header)
    print("-" * len(header))
    for name, true_v, init_v, opt_v, abs_err, rel_err, is_frozen in rows:
        tag = " (frozen)" if is_frozen else ""
        print(
            f"{name:>14s} {true_v:>11.6f} {init_v:>11.6f} {opt_v:>13.6f} "
            f"{abs_err:>+11.6f} {rel_err:>+8.2%}{tag}"
        )

    print()
    if worst_rel <= args.rel_tol:
        print(
            f"PASS: worst relative error {worst_rel:.2%} "
            f"<= tolerance {args.rel_tol:.0%}"
        )
        sys.exit(0)
    else:
        print(
            f"FAIL: worst relative error {worst_rel:.2%} "
            f"> tolerance {args.rel_tol:.0%}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
