import time

import numpy as np

import DTLearner as dt
import LinRegLearner as lrl
from gen_data import best_4_dt, best_4_lin_reg

MAX_TESTS = 15
NEEDED_WINS = 10
SEEDS = [1489683274] + list(range(1, 30))


def count_wins(data_x, data_y, better_cls, worse_cls):
    n = data_x.shape[0]
    cutoff = int(n * 0.6)
    wins = 0
    for _ in range(MAX_TESTS):
        perm = np.random.permutation(n)
        tr, te = perm[:cutoff], perm[cutoff:]
        better, worse = better_cls(), worse_cls()  # grader uses no-arg constructors
        better.add_evidence(data_x[tr], data_y[tr])
        worse.add_evidence(data_x[tr], data_y[tr])
        better_err = np.linalg.norm(data_y[te] - better.query(data_x[te]))
        worse_err = np.linalg.norm(data_y[te] - worse.query(data_x[te]))
        if better_err < 0.9 * worse_err:
            wins += 1
    return wins


def check(gen_fn, better_cls, worse_cls, name):
    failures = []
    for s in SEEDS:
        problems = []

        start = time.time()
        x, y = gen_fn(seed=s)
        elapsed = time.time() - start
        x2, y2 = gen_fn(seed=s)
        x3, y3 = gen_fn(seed=s + 1)

        if elapsed >= 5:
            problems.append(f"took {elapsed:.1f}s")
        if not (np.array_equal(x, x2) and np.array_equal(y, y2)):
            problems.append("same seed gave different data")
        if np.array_equal(x, x3) and np.array_equal(y, y3):
            problems.append("seed + 1 gave identical data")
        if not (10 <= x.shape[0] <= 1000):
            problems.append(f"rows = {x.shape[0]}")
        if x.ndim != 2 or not (2 <= x.shape[1] <= 10):
            problems.append(f"x shape = {x.shape}")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            problems.append(f"y shape = {y.shape} (must be 1-D, length = rows)")
        elif not np.issubdtype(y.dtype, np.floating):
            problems.append(f"y dtype = {y.dtype} (should be float)")
        elif np.all(y == y[0]):
            problems.append("y is constant")

        if not problems or all("took" in p for p in problems):
            wins = count_wins(x, y, better_cls, worse_cls)
            if wins < NEEDED_WINS:
                problems.append(f"only {wins}/{MAX_TESTS} wins (need {NEEDED_WINS})")

        if problems:
            failures.append((s, problems))

    print(f"{name}: {len(SEEDS) - len(failures)}/{len(SEEDS)} seeds passed")
    for s, problems in failures:
        print(f"   seed {s}: {'; '.join(problems)}")


if __name__ == "__main__":
    check(best_4_lin_reg, lrl.LinRegLearner, dt.DTLearner, "best_4_lin_reg")
    check(best_4_dt, dt.DTLearner, lrl.LinRegLearner, "best_4_dt")

    x_lr, _ = best_4_lin_reg(seed=1)
    x_dt, _ = best_4_dt(seed=1)
    if x_lr.shape[0] != x_dt.shape[0]:
        print(f"Row counts differ between functions: {x_lr.shape[0]} vs {x_dt.shape[0]}")