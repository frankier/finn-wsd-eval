import click
from dataclasses import dataclass
from typing import Callable
from plumbum import local
from plumbum.cmd import python, java
from os import makedirs
from os.path import join as pjoin, basename, exists
import sys
import baselines


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
    def run(corpus_fn, truetag_fn, guess_fn):
        all_args = ["baselines.py"] + list(args) + [corpus_fn, guess_fn]
        python(*all_args)
    return run


def ukb(*variant):
    from ukb import run_inner as run_ukb
    def run(corpus_fn, truetag_fn, guess_fn):
        run_ukb(corpus_fn, guess_fn, variant, "ukb-eval/wn30/wn30g.bin", "ukb-eval/wn30/wn30_dict.txt")
    return run


def ims(corpus_fn, truetag_fn, guess_fn):
    from ims import test as ims_test, train as ims_train
    ims_model_path = local.env["IMS_MODEL_PATH"]
    if not exists(ims_model_path):
        train_corpus = local.env["TRAIN_CORPUS"]
        train_corpus_key = local.env["TRAIN_CORPUS_KEY"]
        ims_train.callback(train_corpus, train_corpus_key, ims_model_path)

    ims_test.callback(ims_model_path, corpus_fn, guess_fn)


def ctx2vec(ctx2vec_model):
    def run(corpus_fn, truetag_fn, guess_fn):
        from ctx2vec import test as ctx2vec_test
        ctx2vec_model_path = local.env["CTX2VEC_MODEL_PATH"]
        full_model_path = pjoin(ctx2vec_model_path, ctx2vec_model, "model.params")
        train_corpus = local.env["TRAIN_CORPUS"]
        train_corpus_key = local.env["TRAIN_CORPUS_KEY"]

        ctx2vec_test.callback(full_model_path, train_corpus, train_corpus_key, corpus_fn, guess_fn)
    return run


EXPERIMENTS = [
    Exp("Baseline", None, "first", "FiWN 1st sense", baseline("first")),
    Exp("Baseline", None, "mfe", "FiWN + PWN 1st sense", baseline("mfe")),
    Exp("Supervised", "IMS", "ims", "IMS", ims),
    Exp("Supervised", "Context2Vec", "ctx2vec.noseg.b100", "Context2Vec\\textsubscript{noseg}", ctx2vec("model_noseg_b100")),
    Exp("Supervised", "Context2Vec", "ctx2vec.seg.b100", "Context2Vec\\textsubscript{seg}", ctx2vec("model_seg_b100")),
]

for vec in ["fastText", "numberbatch", "double"]:
    lower_vec = vec.lower()
    for mean in baselines.MEANS.keys():
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
@click.argument("truetag", type=click.Path())
@click.argument("filter_l1", required=False)
@click.argument("filter_l2", required=False)
def main(corpus, truetag, filter_l1=None, filter_l2=None):
    print(TABLE_HEAD) 
    prev_cat = None
    makedirs('guess', exist_ok=True)
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

        guess_fn = "{}.{}.key".format(basename(corpus), exp.nick)
        guess_path = pjoin('guess', guess_fn)

        exp.run(corpus, truetag, guess_path)

        scorer = java["Scorer", truetag, guess_path]
        score_out = scorer()
        print(" & ".join((line.split()[1] for line in score_out.split('\n') if line)), end=" \\\\\n")
    print(TABLE_FOOT)


if __name__ == '__main__':
    main()
