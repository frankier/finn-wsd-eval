import time
import click
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, Optional
from plumbum import local
from plumbum.cmd import python, java
from os import makedirs
from os.path import abspath, join as pjoin, basename, exists
from stiff.eval import get_eval_paths
import shutil
from wsdeval.means import ALL_MEANS, NON_EXPANDING_MEANS, MEAN_DISPS
from tinydb import TinyDB
from tinyrecord import transaction
import os


@dataclass(frozen=True)
class Exp:
    category: str
    subcat: str
    nick: str
    disp: str
    run_func: Optional[Callable[[str, str, str], None]] = None
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
        return info

    def run(self, *args, **kwargs):
        return self.run_func(*args, **kwargs)

    def get_iden(self, corpus):
        return mk_iden(corpus, self)

    def get_paths(self, corpus):
        root, paths = get_eval_paths(corpus)
        iden = self.get_iden(corpus)
        guess_path = mk_guess_path(iden)
        model_path = mk_model_path(iden)
        if self.lex_group:
            gold = paths["test"]["supkey"]
        else:
            gold = paths["test"]["unikey"]
        return paths, guess_path, model_path, gold

    def run_exp(self, db, corpus):
        paths, guess_path, model_path, gold = self.get_paths(corpus)
        try:
            if self.needs_model:
                self.run(paths, guess_path, model_path)
            else:
                self.run(paths, guess_path)
        except Exception:
            import traceback

            traceback.print_exc()
            return
        return self.proc_score(db, corpus, gold, guess_path)

    def proc_score(self, db, corpus, gold, guess):
        measures = score(gold, guess)

        result = self.info()
        result.update(measures)
        result["corpus"] = corpus
        result["time"] = time.time()

        with transaction(db) as tr:
            tr.insert(result)
        return measures


def mk_nick(*inbits):
    outbits = []
    for bit in inbits:
        if isinstance(bit, str):
            outbits.append(bit)
        elif bit is None:
            continue
        elif isinstance(bit, tuple):
            val, fmt = bit
            if isinstance(val, bool):
                if isinstance(fmt, str):
                    outbits.append(fmt if val else "no" + fmt)
                else:
                    outbits.append(fmt[val])
            else:
                assert False
        else:
            assert False
    return ".".join(outbits)


class SupWSD(Exp):
    def __init__(self, vec, sur_words):
        vec_path = "support/emb/{}.txt".format(vec) if vec is not None else ""
        no_sur = "-s" if not sur_words else ""
        disp = f"SupWSD\\textsubscript{{{vec}{no_sur}}}"

        self.vec_path = vec_path
        self.use_vec = vec is not None
        self.sur_words = sur_words
        super().__init__(
            "Supervised",
            "SupWSD",
            mk_nick("supwsd", vec, (sur_words, "sur")),
            disp,
            None,
            {"vec": vec, "sur_words": sur_words},
            needs_model=True,
        )

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.supwsd import conf, train, test
        from wsdeval.utils import proc_supwsd

        if exists(model_path):
            timestr = datetime.now().isoformat()
            shutil.move(model_path, "{}.{}".format(model_path, timestr))
        makedirs(model_path, exist_ok=True)
        conf.callback(
            work_dir=abspath(model_path),
            vec_path=abspath(self.vec_path),
            use_vec=self.use_vec,
            use_surrounding_words=self.sur_words,
        )
        train.callback(paths["train"]["suptag"], paths["train"]["supkey"])
        test.callback(paths["test"]["suptag"], paths["test"]["supkey"])
        with open(paths["test"]["unikey"]) as goldkey, open(
            pjoin(model_path, "scores/plain.result"), "rb"
        ) as supwsd_result_fp, open(guess_fn, "w") as guess_fp:
            proc_supwsd(goldkey, supwsd_result_fp, guess_fp)

    def run_exp(self, db, corpus):
        self.corpus = corpus
        return super().run_exp(db, corpus)


class Elmo(Exp):
    def __init__(self, layer):
        self.layer = layer

        super().__init__(
            "Supervised",
            "ELMo-NN",
            f"elmo_nn.{layer}",
            "ELMO-NN ({})".format(layer),
            None,
            {"layer": layer},
            needs_model=True,
        )

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.elmo import train, test

        with open(paths["train"]["sup"], "rb") as inf, open(
            paths["train"]["supkey"], "r"
        ) as keyin:
            train.callback(inf, keyin, model_path, self.layer)
        with open(paths["test"]["sup"], "rb") as inf, open(guess_fn, "w") as keyout:
            test.callback(model_path, inf, keyout, self.layer)


class ExpGroup:
    def __init__(self, exps):
        self.exps = exps

    def filter_exps(self, filter_l1, filter_l2, opt_dict):
        return [
            exp
            for exp in self.exps
            if self.exp_included(exp, filter_l1, filter_l2, opt_dict)
        ]

    def exp_included(self, exp, filter_l1, filter_l2, opt_dict):
        return (
            (filter_l1 is None or exp.category == filter_l1)
            and (filter_l2 is None or exp.subcat == filter_l2)
            and (
                not opt_dict
                or all((exp.opts[opt] == opt_dict[opt] for opt in opt_dict))
            )
        )

    def run_all(self, db, corpus, filter_l1, filter_l2, opt_dict):
        for exp in self.filter_exps(filter_l1, filter_l2, opt_dict):
            print("Running", exp)
            measures = exp.run_exp(db, corpus)
            print("Got", measures)


class ElmoAllExpGroup(ExpGroup):
    LAYERS = [-1, 0, 1, 2]

    def __init__(self):
        super().__init__([Elmo(layer) for layer in self.LAYERS])

    def run_all(self, db, corpus, filter_l1, filter_l2, opt_dict):
        from wsdeval.systems.elmo import train_all, test_all

        model_paths = []
        guess_paths = []
        keyouts = []
        golds = []
        included = []

        for exp in self.exps:
            if self.exp_included(exp, filter_l1, filter_l2, opt_dict):
                paths, guess_path, model_path, gold = exp.get_paths(corpus)
                included.append(True)
            else:
                model_path = "/dev/null"
                guess_path = "/dev/null"
                included.append(False)
            model_paths.append(model_path)
            guess_paths.append(guess_path)
            keyouts.append(open(guess_path, "w"))
            golds.append(gold)

        print("Running ELMO all")
        try:
            with open(paths["train"]["sup"], "rb") as inf, open(
                paths["train"]["supkey"], "r"
            ) as keyin:
                train_all.callback(inf, keyin, model_paths)
            with open(paths["test"]["sup"], "rb") as inf:
                test_all.callback(inf, zip(model_paths, keyouts))
        except Exception:
            import traceback

            traceback.print_exc()
            return

        for keyout in keyouts:
            keyout.close()

        for exp, gold, guess_path in zip(self.exps, golds, guess_paths):
            print("Measuring", exp)
            measures = exp.proc_score(db, corpus, gold, guess_path)
            print("Got", measures)


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


def ukb(use_new_dict, extract_extra, *variant):
    from wsdeval.systems.ukb import run_inner as run_ukb

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
            extract_extra,
        )

    return run


def ctx2vec(ctx2vec_model, seg):
    def run(paths, guess_fn):
        from wsdeval.systems.ctx2vec import test as ctx2vec_test

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


def nn(vec, mean):
    def inner(paths, guess_fn, model_path):
        from wsdeval.systems.nn import train, test

        with open(paths["train"]["suptag"], "rb") as inf, open(
            paths["train"]["supkey"], "r"
        ) as keyin:
            train.callback(vec, mean, inf, keyin, model_path)
        with open(paths["test"]["suptag"], "rb") as inf, open(guess_fn, "w") as keyout:
            test.callback(vec, mean, model_path, inf, keyout)

    return inner


def lesk_pp(mean, do_expand, exclude_cand, score_by):
    def inner(paths, guess_fn):
        args = [
            "lesk_pp.py",
            mean,
            paths["test"]["unified"],
            guess_fn,
            "--include-wfs",
            "--score-by",
            score_by,
        ]
        if do_expand:
            args.append("--expand")
        if exclude_cand:
            args.append("--exclude-cand")
        python(*args)

    return inner


EXPERIMENTS = [
    ExpGroup(
        [
            Exp("Baseline", None, "first", "FiWN 1st sense", baseline("first")),
            Exp("Baseline", None, "mfe", "FiWN + PWN 1st sense", baseline("mfe")),
        ]
    ),
    ExpGroup(
        [
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
    ),
]


supwsd_exps = []
for vec, sur_words in [
    (None, True),
    ("word2vec", False),
    ("word2vec", True),
    ("fasttext", False),
    ("fasttext", True),
]:
    supwsd_exps.append(SupWSD(vec, sur_words))
EXPERIMENTS.append(ExpGroup(supwsd_exps))


LESK_MEANS = list(ALL_MEANS.keys())
LESK_MEANS.remove("normalized_mean")
LESK_MEANS.append("sif_mean")


xlingual_lesk = []
for use_freq in [False, True]:
    for do_expand in [False, True]:
        for vec in ["fasttext", "numberbatch", "double"]:
            lower_vec = vec.lower()
            for mean in LESK_MEANS:
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
                    if use_freq:
                        baseline_args += ["--use-freq"]
                        nick_extra += ".freq"
                        disp_extra += "+freq"
                    nick = f"lesk.{lower_vec}.{mean}{nick_extra}"
                    mean_disp = "+" + MEAN_DISPS[mean]
                    disp = f"Lesk\\textsubscript{{{vec}{disp_extra}{mean_disp}}}"
                    xlingual_lesk.append(
                        Exp(
                            "Knowledge",
                            "Cross-lingual Lesk",
                            nick,
                            disp,
                            lesk(*baseline_args),
                            {
                                "vec": vec,
                                "expand": do_expand,
                                "mean": mean,
                                "wn_filter": wn_filter,
                                "use_freq": use_freq,
                            },
                        )
                    )
EXPERIMENTS.append(ExpGroup(xlingual_lesk))

awe_nn_exps = []
for vec in ["fasttext", "word2vec", "numberbatch", "triple", "double"]:
    for mean in list(ALL_MEANS.keys()) + ["sif_mean"]:
        awe_nn_exps.append(
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
EXPERIMENTS.append(ExpGroup(awe_nn_exps))


if os.environ.get("USE_SINGLE_LAYER_ELMO"):
    elmo_exps = []
    for layer in (-1, 0, 1, 2):
        elmo_exps.append(Elmo(layer))
    EXPERIMENTS.append(ExpGroup(elmo_exps))
else:
    EXPERIMENTS.append(ElmoAllExpGroup())


lesk_pp_exps = []
for score_by in ["both", "defn", "lemma"]:
    for exclude_cand in [False, True]:
        for do_expand in [False, True]:
            for mean in NON_EXPANDING_MEANS:
                lesk_pp_exps.append(
                    Exp(
                        "Knowledge",
                        "Lesk++",
                        "lesk_pp",
                        "Lesk++ ({} {})".format(MEAN_DISPS[mean], do_expand),
                        lesk_pp(mean, do_expand, exclude_cand, score_by),
                        {
                            "mean": mean,
                            "expand": do_expand,
                            "exclude_cand": exclude_cand,
                            "score_by": score_by,
                        },
                    )
                )
EXPERIMENTS.append(ExpGroup(lesk_pp_exps))


ukb_exps = []
for use_freq in [False, True]:
    for extract_extra in [False, True]:
        ukb_args = ("--ppr_w2w",)
        label_extra = ""
        nick_extra = ""
        if use_freq:
            ukb_args += ("--dict_weight",)
        else:
            ukb_args += ("--dict_noweight",)
            label_extra = "_nf"
            nick_extra += ".nf"
        if extract_extra:
            label_extra += "+extract"
            nick_extra += ".extract"
        ukb_exps.append(
            Exp(
                "Knowledge",
                "UKB",
                "ukb" + nick_extra,
                "UKB{}".format(label_extra),
                ukb(True, extract_extra, *ukb_args),
                {"extract_extra": extract_extra, "use_freq": use_freq},
            )
        )
EXPERIMENTS.append(ExpGroup(ukb_exps))


def score(gold, guess):
    scorer = java["Scorer", gold, guess]
    score_out = scorer()
    measures = {}
    for line in score_out.split("\n"):
        if not line:
            continue
        bits = line.split()
        assert bits[0][-1] == "="
        measures[bits[0][:-1]] = bits[1]
    return measures


def mk_guess_path(iden):
    guess_fn = iden + ".key"
    return pjoin("guess", guess_fn)


def mk_model_path(iden):
    timestr = datetime.now().isoformat()
    return pjoin("models", f"{iden}.{timestr}")


def mk_iden(corpus, exp):
    corpus_basename = basename(corpus.rstrip("/"))
    return "{}.{}".format(corpus_basename, exp.nick)


def parse_opts(opts):
    opt_dict = {}
    for opt in opts:
        k, v = opt.split("=")
        if v in ["True", "False"]:
            py_v = v == "True"
        elif v == "None":
            py_v = None
        else:
            try:
                py_v = int(v)
            except ValueError:
                py_v = v
        opt_dict[k] = py_v
    return opt_dict


@click.command()
@click.argument("db_path", type=click.Path())
@click.argument("corpus", type=click.Path())
@click.argument("filter_l1", required=False)
@click.argument("filter_l2", required=False)
@click.argument("opts", nargs=-1)
def main(db_path, corpus, filter_l1=None, filter_l2=None, opts=None):
    makedirs("guess", exist_ok=True)
    makedirs("models", exist_ok=True)
    db = TinyDB(db_path).table("results")
    if opts:
        opt_dict = parse_opts(opts)
    else:
        opt_dict = {}
    for exp_group in EXPERIMENTS:
        exp_group.run_all(db, corpus, filter_l1, filter_l2, opt_dict)


if __name__ == "__main__":
    main()
