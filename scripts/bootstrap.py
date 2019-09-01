import pickle
from wsdeval.exps.utils import score
import click
from memory_tempfile import MemoryTempfile
import random

tempfile = MemoryTempfile()


@click.group()
def bootstrap():
    pass


@bootstrap.command("pair")
@click.argument("gold", type=click.Path())
@click.argument("guess_a", type=click.Path())
@click.argument("guess_b", type=click.Path())
def paired_bootstrap_cmd(gold, guess_a, guess_b):
    b_bigger, p_val = paired_bootstrap(gold, guess_a, guess_b)
    if b_bigger:
        print(f"{guess_a} < {guess_b}")
    else:
        print(f"{guess_a} < {guess_b}")
    print(f"P = {p_val}")


@bootstrap.command("all-pairs")
@click.argument("dumpf", type=click.File("wb"))
@click.argument("gold", type=click.Path())
@click.argument("guesses", type=click.Path(), nargs=-1)
def all_pairs_bootstrap_cmd(dumpf, gold, guesses):
    """
    Compares all pairs the slow way. Usually it would be better to use
    resample-f1s and compare-f1s.
    """
    result = all_pairs_bootstrap(gold, guesses)
    pickle.dump((result, guesses), dumpf)


@bootstrap.command("resample-f1s")
@click.argument("dumpf", type=click.File("wb"))
@click.argument("gold", type=click.Path())
@click.argument("guesses", type=click.Path(), nargs=-1)
@click.option("--iters", type=int, default=1000)
@click.option("--seed", type=int, default=None)
def resample_f1s_cmd(dumpf, gold, guesses, iters, seed):
    """
    Get many f1s from resampled versions of the corpus.
    """
    result = resample_f1s(gold, guesses, bootstrap_iters=iters, seed=seed)
    pickle.dump((result, guesses), dumpf)


@bootstrap.command("compare-f1s")
@click.argument("inf", type=click.File("rb"))
@click.argument("outf", type=click.File("wb"))
def compare_f1s_cmd(inf, outf):
    (orig_f1s, resampled_f1s), guesses = pickle.load(inf)
    result = compare_f1s(orig_f1s, resampled_f1s)
    pickle.dump((result, guesses), outf)


@bootstrap.command("disp-cmp")
@click.argument("inf", type=click.File("rb"))
def disp_cmp(inf):
    result, guesses = pickle.load(inf)
    print("guesses", guesses)
    print("result", result)


def get_f1(score_dict):
    return float(score_dict["F1"].rstrip("%"))


def mk_bootstrap_schedule(bootstrap_size, bootstrap_iters=1000, seed=None):
    if seed is not None:
        random.seed(seed)
    res = []
    for _ in range(bootstrap_iters):
        resample = []
        for _ in range(bootstrap_size):
            resample.append(random.randrange(bootstrap_size))
        res.append(resample)
    return res


def bootstrap_f1s(gold, guess, schedule):
    guess_lines = open(guess).readlines()

    f1s = []
    boot = tempfile.NamedTemporaryFile("w+")
    for resample in schedule:
        boot.seek(0)
        boot.truncate()
        for sample_idx in resample:
            boot.write(guess_lines[sample_idx])
        boot.flush()
        f1s.append(get_f1(score(gold, boot.name)))
    return f1s


def pair_f1s(orig_f1_a, orig_f1_b, f1s_a, f1s_b):
    sample_diff = orig_f1_b - orig_f1_a
    if sample_diff < 0:
        sample_diff = -sample_diff
        b_bigger = False
    else:
        b_bigger = True
    s = 0
    for f1_a, f1_b in zip(f1s_a, f1s_b):
        if b_bigger:
            resamped_diff = f1_b - f1_a
        else:
            resamped_diff = f1_a - f1_b
        if resamped_diff > 2 * sample_diff:
            s += 1
    return b_bigger, s / len(f1s_a)


def paired_bootstrap(gold, guess_a, guess_b, bootstrap_size=None, bootstrap_iters=1000):
    sample_mean_a = get_f1(score(gold, guess_a))
    sample_mean_b = get_f1(score(gold, guess_b))
    sample_diff = sample_mean_b - sample_mean_a
    if sample_diff < 0:
        sample_diff = -sample_diff
        b_bigger = False
    else:
        b_bigger = True
    guess_a_lines = open(guess_a).readlines()
    guess_b_lines = open(guess_b).readlines()
    assert len(guess_a_lines) == len(guess_a_lines)
    paired_lines = list(zip(guess_a_lines, guess_b_lines))
    if bootstrap_size is None:
        bootstrap_size = len(guess_a_lines)

    boot_a = tempfile.NamedTemporaryFile("w+")
    boot_b = tempfile.NamedTemporaryFile("w+")
    s = 0
    for iter_idx in range(bootstrap_iters):
        boot_a.seek(0)
        boot_a.truncate()
        boot_b.seek(0)
        boot_b.truncate()
        resampled = random.choices(paired_lines, k=bootstrap_size)
        for line_a, line_b in resampled:
            boot_a.write(line_a)
            boot_b.write(line_b)
        boot_a.flush()
        boot_b.flush()
        resampled_mean_a = get_f1(score(gold, boot_a.name))
        resampled_mean_b = get_f1(score(gold, boot_b.name))
        if b_bigger:
            resamped_diff = resampled_mean_b - resampled_mean_a
        else:
            resamped_diff = resampled_mean_a - resampled_mean_b
        if resamped_diff > 2 * sample_diff:
            s += 1
    return b_bigger, s / bootstrap_iters


class IterPairs:
    def __init__(self, guesses):
        self.guesses = guesses

    def __len__(self):
        num_guesses = len(self.guesses)
        res = num_guesses * (num_guesses - 1) // 2
        return res

    def __iter__(self):
        for guess_idx, guess_a in enumerate(self.guesses):
            for guess_b in self.guesses[guess_idx + 1 :]:
                yield guess_idx, guess_a, guess_b


def resample_f1s(gold, guesses, bootstrap_iters=1000, seed=None):
    gold_len = len(open(gold).readlines())
    schedule = mk_bootstrap_schedule(
        gold_len, bootstrap_iters=bootstrap_iters, seed=seed
    )
    orig_f1s = []
    resampled_f1s = []
    guesses_ctx = click.progressbar(
        guesses, label="Getting resampled F1s", show_pos=True
    )
    with guesses_ctx as guesses_it:
        for guess in guesses_it:
            assert (
                len(open(guess).readlines()) == gold_len
            ), f"Gold {gold} is not same length as {guess}"
            orig_f1s.append(get_f1(score(gold, guess)))
            resampled_f1s.append(bootstrap_f1s(gold, guess, schedule))
    return orig_f1s, resampled_f1s


def compare_f1s(orig_f1s, resampled_f1s):
    iter_pairs = IterPairs(list(zip(orig_f1s, resampled_f1s)))
    pairs_ctx = click.progressbar(iter_pairs, label="Comparing pairs", show_pos=True)
    result = [[] for _ in orig_f1s]
    with pairs_ctx as pairs:
        for idx, (orig_f1_a, f1s_a), (orig_f1_b, f1s_b) in pairs:
            b_bigger, p_val = pair_f1s(orig_f1_a, orig_f1_b, f1s_a, f1s_b)
            result[idx].append((b_bigger, p_val))
    return result


def all_pairs_bootstrap(gold, guesses):
    result = [[] for _ in guesses]
    iter_pairs = IterPairs(guesses)
    iter_ctx = click.progressbar(
        iter_pairs, label="Bootstrapping all pairs", show_pos=True
    )
    with iter_ctx as iter:
        for guess_a_idx, guess_a, guess_b in iter:
            b_bigger, p_val = paired_bootstrap(gold, guess_a, guess_b)
            result[guess_a_idx].append((b_bigger, p_val))
    return result


if __name__ == "__main__":
    bootstrap()
