import click
from dataclasses import dataclass
from typing import Callable
from plumbum import local
from plumbum.cmd import python, java
from os import makedirs
from os.path import join as pjoin, basename


@dataclass(frozen=True)
class Exp:
    category: str
    nick: str
    disp: str
    run: Callable[[str, str, str], None]


def baseline(*args):
    def run(corpus_fn, truetag_fn, guess_fn):
        all_args = ["baselines.py"] + list(args) + [corpus_fn, guess_fn]
        #print(" ".join(all_args))
        python(*all_args)
    return run


def ukb(*variant):
    from ukb import run_inner as run_ukb
    def run(corpus_fn, truetag_fn, guess_fn):
        run_ukb(corpus_fn, guess_fn, variant, "ukb-eval/wn30/wn30g.bin", "ukb-eval/wn30/wn30_dict.txt")
    return run


def ims(corpus_fn, truetag_fn, guess_fn):
    from ims import test as ims_test
    ims_model = local.env["IMS_MODEL"]
    ims_test.callback(ims_model, corpus_fn, guess_fn)


def ctx2vec(ctx2vec_model):
    from ctx2vec import test as ctx2vec_test
    ctx2vec_model_path = local.env["CTX2VEC_MODEL_PATH"]
    full_model_path = pjoin(ctx2vec_model_path, ctx2vec_model, "model.params")
    ctx2vec_train_corpus = local.env["CTX2VEC_TRAIN_CORPUS"]
    ctx2vec_train_corpus_key = local.env["CTX2VEC_TRAIN_CORPUS_KEY"]
    def run(corpus_fn, truetag_fn, guess_fn):
        ctx2vec_test.callback(full_model_path, ctx2vec_train_corpus, ctx2vec_train_corpus_key, corpus_fn, guess_fn)
    return run


EXPERIMENTS = [
    Exp("Baseline", "first", "FiWN 1st sense", baseline("first")),
    Exp("Baseline", "mfe", "FiWN + PWN 1st sense", baseline("mfe")),
    Exp("Baseline", "lesk.fasttext", "Lesk\\textsubscript{fastText+AWE}", baseline("lesk_fasttext")),
    Exp("Baseline", "lesk.fasttext.wn-filter", "Lesk\\textsubscript{fastText+AWE+WN-filter}", baseline("lesk_fasttext", "--wn-filter")),
    Exp("Baseline", "lesk.numberbatch", "Lesk\\textsubscript{numberbatch+AWE}", baseline("lesk_conceptnet")),
    Exp("Baseline", "lesk.numberbatch.wn-filter", "Lesk\\textsubscript{numberbatch+AWE+WN-filter}", baseline("lesk_conceptnet", "--wn-filter")),
    Exp("Supervised", "ims", "IMS", ims),
    Exp("Supervised", "ctx2vec.noseg.b100", "Context2Vec\\textsubscript{noseg}", ctx2vec("model_noseg_b100")),
    Exp("Supervised", "ctx2vec.seg.b100", "Context2Vec\\textsubscript{seg}", ctx2vec("model_seg_b100")),
]

# XXX: default configuration -- possibly bad due to using 1st sense + bad data
DICTABLE_OPTS = [("--ppr_w2w",), ("--ppr",), ("--dgraph_dfs", "--dgraph_rank", "ppr")]
VARIANTS = []
for extra in [(), ("--nodict_weight",)]:
    for opt in DICTABLE_OPTS:
        VARIANTS.append(opt + extra)
VARIANTS.append(("--static",))

for idx, variant in enumerate(VARIANTS):
    EXPERIMENTS.append(Exp("Knowledge", "UKB", "UKB{}".format(idx), ukb(*variant)))

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
def main(corpus, truetag):
    print(TABLE_HEAD) 
    prev_cat = None
    makedirs('guess', exist_ok=True)
    for exp in EXPERIMENTS:
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
