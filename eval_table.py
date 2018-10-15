import time
import click
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict
from plumbum import local
from plumbum.cmd import python, java
from os import makedirs
from os.path import abspath, join as pjoin, basename, exists
from stiff.eval import get_eval_paths
import sys
import shutil
import baselines
from means import ALL_MEANS, NON_EXPANDING_MEANS, MEAN_DISPS
from tinydb import TinyDB, where
from tinyrecord import transaction


@dataclass(frozen=True)
class Exp:
    category: str
    subcat: str
    nick: str
    disp: str
    run: Callable[[str, str, str], None]
    opts: Dict[str, any] = field(default_factory=dict)
    lex_group: bool = False
    needs_model: bool = False

    def info(self):
        info = {
            "category": self.category,
            "subcat": self.subcat,
            "nick": self.nick,
            "disp": self.disp,
        }
        info.update(self.opts)


def baseline(*args):
    def run(paths, guess_fn):
        all_args = ["baselines.py"] + list(args) + [paths["test"]["unified"], guess_fn]
        python(*all_args)

    return run


def lesk(variety, *args):
    def run(paths, guess_fn):
        all_args = (
            ["lesk.py", variety] + list(args) + [paths["test"]["suptag"], guess_fn]
        )
        python(*all_args)

    return run


def ukb(use_new_dict, *variant):
    from ukb import run_inner as run_ukb

    def run(paths, guess_fn):
        if use_new_dict:
            dict_fn = "support/ukb/wndict.fi.txt"
        else:
            dict_fn = "support/ukb/wn30/wn30_dict.txt"
        run_ukb(
            paths["test"]["unified"],
            guess_fn,
            variant,
            "support/ukb/wn30/wn30g.bin",
            dict_fn,
        )

    return run


def ctx2vec(ctx2vec_model, seg):
    def run(paths, guess_fn):
        from ctx2vec import test as ctx2vec_test

        ctx2vec_model_path = local.env["CTX2VEC_MODEL_PATH"]
        full_model_path = pjoin(ctx2vec_model_path, ctx2vec_model, "model.params")
        if seg:
            train_corpus = paths["train"]["supseg"]
            test_corpus = paths["test"]["supseg"]
        else:
            train_corpus = paths["train"]["sup"]
            test_corpus = paths["test"]["sup"]

        ctx2vec_test.callback(
            full_model_path,
            train_corpus,
            paths["train"]["sup3key"],
            test_corpus,
            paths["test"]["sup3key"],
            guess_fn,
        )

    return run


def supwsd(vec_path, use_vec, use_surrounding_words):
    def run(paths, guess_fn, model_path):
        from supwsd import conf, train, test
        from utils import iter_supwsd_result

        if exists(model_path):
            timestr = datetime.now().isoformat()
            shutil.move(model_path, "{}.{}".format(model_path, timestr))
        makedirs(model_path, exist_ok=True)
        conf.callback(
            work_dir=abspath(model_path),
            vec_path=abspath(vec_path),
            use_vec=use_vec,
            use_surrounding_words=use_surrounding_words,
        )
        train.callback(paths["train"]["suptag"], paths["train"]["supkey"])
        test.callback(paths["test"]["suptag"], paths["test"]["supkey"])
        with open(paths["test"]["unikey"]) as goldkey, open(
            pjoin(model_path, "scores/plain.result")
        ) as supwsd_result_fp, open(guess_fn, "w") as guess_fp:
            for gold_line, supwsd_result in zip(
                goldkey, iter_supwsd_result(supwsd_result_fp)
            ):
                key = gold_line.split()[0]
                synset = supwsd_result[1][0][0]
                guess_fp.write("{} {}\n".format(key, synset))

    return run


def elmo(layer):
    def run(paths, guess_fn):
        from elmo import train, test

        model = "models/elmo." + str(layer)
        with open(paths["train"]["sup"], "rb") as inf, open(
            paths["train"]["supkey"], "r"
        ) as keyin, open(model, "wb") as modelout:
            train.callback(inf, keyin, modelout, layer)
        with open(model, "rb") as modelin, open(
            paths["test"]["sup"], "rb"
        ) as inf, open(guess_fn, "w") as keyout:
            test.callback(modelin, inf, keyout, layer)

    return run


def nn(vec, mean):
    def inner(paths, guess_fn, model_path):
        from nn import train, test

        with open(paths["train"]["suptag"], "rb") as inf, open(
            paths["train"]["supkey"], "r"
        ) as keyin, open(model_path, "wb") as modelout:
            train.callback(vec, mean, inf, keyin, modelout)
        with open(model_path, "rb") as modelin, open(
            paths["test"]["suptag"], "rb"
        ) as inf, open(guess_fn, "w") as keyout:
            test.callback(vec, mean, modelin, inf, keyout)

    return inner


def lesk_pp(mean):
    def inner(paths, guess_fn):
        python("lesk_pp.py", mean, paths["test"]["unified"], guess_fn, "--include-wfs")

    return inner


EXPERIMENTS = [
    Exp("Baseline", None, "first", "FiWN 1st sense", baseline("first")),
    Exp("Baseline", None, "mfe", "FiWN + PWN 1st sense", baseline("mfe")),
    Exp(
        "Supervised",
        "Context2Vec",
        "ctx2vec.noseg.b100",
        "Context2Vec\\textsubscript{noseg}",
        ctx2vec("model_noseg_b100", False),
    ),
    Exp(
        "Supervised",
        "Context2Vec",
        "ctx2vec.seg.b100",
        "Context2Vec\\textsubscript{seg}",
        ctx2vec("model_seg_b100", True),
    ),
]


for vec, sur_words in [
    (None, True),
    ("word2vec", False),
    ("word2vec", True),
    ("fasttext", False),
    ("fasttext", True),
]:
    if vec is not None:
        vec_path = "support/emb/{}.txt".format(vec)
        use_vec = True
        nick_extra = vec
    else:
        vec_path = ""
        use_vec = False
        nick_extra = ""
    no_sur = "-s" if not sur_words else ""
    if sur_words:
        nick_extra += ".sur"
    else:
        nick_extra += ".nosur"
    disp = f"SupWSD\\textsubscript{{{vec}{no_sur}}}"
    EXPERIMENTS.append(
        Exp(
            "Supervised",
            "SupWSD",
            "supwsd." + nick_extra,
            disp,
            supwsd(vec_path, use_vec, sur_words),
            {"vec": vec, "sur_words": sur_words},
            needs_model=True,
        )
    )


LESK_MEANS = ALL_MEANS.copy()
del LESK_MEANS["normalized_mean"]


for do_expand in [False, True]:
    for vec in ["fasttext", "numberbatch", "double"]:
        lower_vec = vec.lower()
        for mean in LESK_MEANS.keys():
            for wn_filter in [False, True]:
                baseline_args = [lower_vec, mean]
                nick_extra = ""
                disp_extra = ""
                if wn_filter:
                    baseline_args += ["--wn-filter"]
                    nick_extra += ".wn-filter"
                    disp_extra += "+WN-filter"
                if do_expand:
                    baseline_args += ["--expand"]
                    nick_extra += ".expand"
                    disp_extra += "+expand"
                nick = "lesk." + lower_vec + nick_extra
                mean_disp = "+" + MEAN_DISPS[mean]
                disp = f"Lesk\\textsubscript{{{vec}{disp_extra}{mean_disp}}}"
                EXPERIMENTS.append(
                    Exp(
                        "Knowledge",
                        "X-lingual Lesk".format(),
                        nick,
                        disp,
                        lesk(*baseline_args),
                        {"vec": vec, "expand": do_expand, "mean": mean},
                    )
                )

for vec in ["fasttext", "word2vec", "numberbatch", "triple"]:
    for mean in ALL_MEANS.keys():
        EXPERIMENTS.append(
            Exp(
                "Supervised",
                "AWE-NN",
                "awe_nn",
                "AWE-NN ({}, {})".format(vec, MEAN_DISPS[mean]),
                nn(vec, mean),
                {"vec": vec, "mean": mean},
                needs_model=True,
            )
        )

for layer in (-1, 0, 1, 2):
    EXPERIMENTS.append(
        Exp(
            "Supervised",
            "ELMo-NN",
            "elmo_nn",
            "ELMO-NN ({})".format(layer),
            {"layer": layer},
            elmo(layer),
        )
    )

for mean in NON_EXPANDING_MEANS.keys():
    EXPERIMENTS.append(
        Exp(
            "Knowledge",
            "Lesk++",
            "lesk_pp",
            "Lesk++ ({})".format(MEAN_DISPS[mean]),
            {"mean": mean},
            lesk_pp(mean),
        )
    )

UKB_VARIANTS = []
for extra in [("--dict_weight",), ("--dict_noweight",)]:
    UKB_VARIANTS.append(("--ppr_w2w",) + extra)

UKB_LABELS = ["", "_nf"]

for idx, variant in enumerate(UKB_VARIANTS):
    EXPERIMENTS.append(
        Exp(
            "Knowledge",
            "UKB",
            "UKB",
            "UKB{}".format(UKB_LABELS[idx]),
            ukb(True, *variant),
        )
    )

TABLE_HEAD = r"""
\begin{tabu} to \linewidth { l X r r r }
  \toprule
   & System & P & R & F$_1$ \\
"""
# \multirow{3}{*}{Baseline} & FiWN 1st sense & 29.2\% & 29.2\% & 29.2\% \\
# & FiWN + PWN 1st sense & 50.3\% & 50.3\% & 50.3\% \\
# & Lesk with fasttext vector averaging & 29.4\% & 29.4\% & 29.4\% \\
# \midrule
# \midrule
# \multirow{2}{*}{Knowledge} & UKB (default configuration -- possibly bad due to using 1st sense + bad data) & 51.3\% & 50.6\% & 50.9\% \\
TABLE_FOOT = r"""
  \bottomrule
\end{tabu}
"""


@click.command()
@click.argument("corpus", type=click.Path())
@click.argument("filter_l1", required=False)
@click.argument("filter_l2", required=False)
def main(corpus, filter_l1=None, filter_l2=None):
    root, paths = get_eval_paths(corpus)
    print(TABLE_HEAD)
    prev_cat = None
    makedirs("guess", exist_ok=True)
    makedirs("models", exist_ok=True)
    table = TinyDB("results.json").table("results")
    for exp in EXPERIMENTS:
        if (filter_l1 is not None and exp.category != filter_l1) or (
            filter_l2 is not None and exp.subcat != filter_l2
        ):
            continue
        if exp.category != prev_cat:
            prev_cat = exp.category
            print("\midrule")
            print(f"\multirow{{3}}{{*}} {exp.category} & ", end="")
        else:
            print(f" & ", end="")
        print(exp.disp, end=" & ")

        corpus_basename = basename(corpus.rstrip("/"))
        iden = "{}.{}".format(corpus_basename, exp.nick)
        guess_fn = iden + ".key"
        guess_path = pjoin("guess", guess_fn)
        model_path = pjoin("models", iden)

        try:
            if exp.needs_model:
                exp.run(paths, guess_path, model_path)
            else:
                exp.run(paths, guess_path)
        except Exception:
            import traceback

            traceback.print_exc()
            continue

        if exp.lex_group:
            gold = paths["test"]["supkey"]
        else:
            gold = paths["test"]["unikey"]
        scorer = java["Scorer", gold, guess_path]
        score_out = scorer()
        measures = {}
        for line in score_out.split("\n"):
            if not line:
                continue
            bits = line.split()
            assert bits[0][-1] == "="
            measures[bits[0][:-1]] = bits[1]
        # print(" & ".join(), end=" \\\\\n")

        result = exp.info()
        result.update(measures)
        result["corpus"] = corpus
        result["time"] = time.time()

        with transaction(table) as tr:
            tr.insert(result)
    print(TABLE_FOOT)


if __name__ == "__main__":
    main()
