import click
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from plumbum import local
from plumbum.cmd import python, java
from os import makedirs
from os.path import join as pjoin, basename, exists
from stiff.eval import get_eval_paths
import sys
import shutil
import baselines
from means import MEANS


MEAN_DISPS = {
    "catp3_mean": "CATP3-WE",
    "catp4_mean": "CATP4-WE",
    "sif_mean": "SIF-WE",
    "unnormalized_mean": "AWE",
    "normalized_mean": "AWE-norm",
}


@dataclass(frozen=True)
class Exp:
    category: str
    subcat: str
    nick: str
    disp: str
    run: Callable[[str, str, str], None]


def baseline(*args):
    def run(paths, guess_fn):
        all_args = ["baselines.py"] + list(args) + [paths["test"]["unified"], guess_fn]
        python(*all_args)
    return run


def ukb(*variant):
    from ukb import run_inner as run_ukb
    def run(paths, guess_fn):
        run_ukb(paths["test"]["unified"], guess_fn, variant, "ukb-eval/wn30/wn30g.bin", "ukb-eval/wn30/wn30_dict.txt")
    return run


def ims(paths, guess_fn):
    from ims import test as ims_test, train as ims_train
    ims_model_path = "models/ims"
    if exists(ims_model_path):
        timestr = datetime.now().isoformat()
        shutil.move(ims_model_path, "{}.{}".format(ims_model_path, timestr))
    makedirs(ims_model_path, exist_ok=True)
    ims_train.callback(paths["train"]["suptag"], paths["train"]["supkey"], ims_model_path)
    ims_test.callback(ims_model_path, paths["test"]["suptag"], guess_fn)


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

        ctx2vec_test.callback(full_model_path, train_corpus, paths["train"]["supkey"], test_corpus, guess_fn)
    return run


def supwsd(paths, guess_fn):
    from supwsd import train, test
    supwsd_model_path = "models/supwsd"
    if exists(supwsd_model_path):
        timestr = datetime.now().isoformat()
        shutil.move(supwsd_model_path, "{}.{}".format(supwsd_model_path, timestr))
    makedirs(supwsd_model_path, exist_ok=True)
    print("train", paths["train"]["suptag"], paths["train"]["supkey"])
    train.callback(paths["train"]["suptag"], paths["train"]["supkey"])
    print("test", paths["test"]["suptag"], guess_fn)
    test.callback(paths["test"]["suptag"], guess_fn)


def lesk_pp(paths, guess_fn):
    python("lesk_pp.py", paths["test"]["unified"], guess_fn, "--include-wfs")


EXPERIMENTS = [
    Exp("Baseline", None, "first", "FiWN 1st sense", baseline("first")),
    Exp("Baseline", None, "mfe", "FiWN + PWN 1st sense", baseline("mfe")),
    #Exp("Supervised", "IMS", "ims", "IMS", ims),
    Exp("Supervised", "Context2Vec", "ctx2vec.noseg.b100", "Context2Vec\\textsubscript{noseg}", ctx2vec("model_noseg_b100", False)),
    Exp("Supervised", "Context2Vec", "ctx2vec.seg.b100", "Context2Vec\\textsubscript{seg}", ctx2vec("model_seg_b100", True)),
    Exp("Supervised", "SupWSD", "supwsd", "SupWSD", supwsd),
    Exp("Knowledge", "Lesk++", "lesk_pp", "Lesk++", lesk_pp),
]

for vec in ["fastText", "numberbatch", "double"]:
    lower_vec = vec.lower()
    for mean in MEANS.keys():
        for wn_filter in [False, True]:
            baseline_args = ["lesk_" + lower_vec, mean]
            if wn_filter:
                baseline_args += ["--wn-filter"]
                nick_extra = ".wn-filter"
                disp_extra = "+WN-filter"
            else:
                nick_extra = ""
                disp_extra = ""
            nick = "lesk." + lower_vec + nick_extra
            disp = f"Lesk\\textsubscript{{{vec}{disp_extra}}}"
            EXPERIMENTS.append(Exp("Knowledge", "Multilingual Word Vector Lesk", nick, disp, baseline(*baseline_args)))

# XXX: default configuration -- possibly bad due to using 1st sense + bad data
DICTABLE_OPTS = [("--ppr_w2w",), ("--ppr",), ("--dgraph_dfs", "--dgraph_rank", "ppr")]
VARIANTS = []
for extra in [(), ("--nodict_weight",)]:
    for opt in DICTABLE_OPTS:
        VARIANTS.append(opt + extra)
VARIANTS.append(("--static",))

for idx, variant in enumerate(VARIANTS):
    EXPERIMENTS.append(Exp("Knowledge", "UKB", "UKB", "UKB{}".format(idx), ukb(*variant)))

TABLE_HEAD = r"""
\begin{tabu} to \linewidth { l X r r r }
  \toprule
   & System & P & R & F$_1$ \\
"""
    #\multirow{3}{*}{Baseline} & FiWN 1st sense & 29.2\% & 29.2\% & 29.2\% \\
     #& FiWN + PWN 1st sense & 50.3\% & 50.3\% & 50.3\% \\
     #& Lesk with fasttext vector averaging & 29.4\% & 29.4\% & 29.4\% \\
    #\midrule
    #Supervised & IMS & & & \\
    #\midrule
    #\multirow{2}{*}{Knowledge} & UKB (default configuration -- possibly bad due to using 1st sense + bad data) & 51.3\% & 50.6\% & 50.9\% \\
TABLE_FOOT = r"""
  \bottomrule
\end{tabu}
"""

@click.command()
@click.argument("corpus", type=click.Path())
@click.argument("filter_l1", required=False)
@click.argument("filter_l2", required=False)
def main(corpus, filter_l1=None, filter_l2=None):
    paths = get_eval_paths(corpus)
    print(TABLE_HEAD) 
    prev_cat = None
    makedirs('guess', exist_ok=True)
    makedirs('models', exist_ok=True)
    for exp in EXPERIMENTS:
        if (filter_l1 is not None and exp.category != filter_l1) or \
                (filter_l2 is not None and exp.subcat != filter_l2):
            continue
        if exp.category != prev_cat:
            prev_cat = exp.category
            print("\midrule")
            print(f"\multirow{{3}}{{*}} {exp.category} & ", end="")
        else:
            print(f" & ", end="")
        print(exp.disp, end=" & ")

        corpus_basename = basename(corpus.rstrip("/"))
        guess_fn = "{}.{}.key".format(corpus_basename, exp.nick)
        guess_path = pjoin('guess', guess_fn)

        try:
            exp.run(paths, guess_path)
        except:
            import traceback
            traceback.print_exc()
            continue

        scorer = java["Scorer", paths["test"]["unikey"], guess_path]
        score_out = scorer()
        print(" & ".join((line.split()[1] for line in score_out.split('\n') if line)), end=" \\\\\n")
    print(TABLE_FOOT)


if __name__ == '__main__':
    main()
