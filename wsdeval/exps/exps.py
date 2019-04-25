import os
from plumbum.cmd import python
from expcomb.models import Exp, SupExp, ExpGroup
from .utils import mk_nick
from stiff.eval import get_partition_paths
import traceback
import datetime
from os import makedirs
from os.path import abspath, exists, join as pjoin
import shutil
from wsdeval.tools.means import MEAN_DISPS

SYSTEMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../systems")


class SupWSD(SupExp):
    def __init__(self, vec, sur_words):
        vec_path = "support/emb/{}.txt".format(vec) if vec is not None else ""
        no_sur = "-s" if not sur_words else ""
        disp = f"SupWSD\\textsubscript{{{vec}{no_sur}}}"

        self.vec_path = vec_path
        self.use_vec = vec is not None
        self.sur_words = sur_words
        super().__init__(
            ["Supervised", "SupWSD"],
            mk_nick("supwsd", vec, (sur_words, "sur")),
            disp,
            None,
            {"vec": vec, "sur_words": sur_words},
        )

    def conf(self, model_path):
        from wsdeval.systems.supwsd import conf

        conf.callback(
            work_dir=abspath(model_path),
            vec_path=abspath(self.vec_path),
            use_vec=self.use_vec,
            use_surrounding_words=self.sur_words,
        )

    def train(self, paths, model_path):
        from wsdeval.systems.supwsd import train

        self.conf(model_path)

        if exists(model_path):
            timestr = datetime.now().isoformat()
            shutil.move(model_path, "{}.{}".format(model_path, timestr))
        makedirs(model_path, exist_ok=True)
        train.callback(paths["suptag"], paths["supkey"])

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.supwsd import test
        from wsdeval.formats.supwsd import proc_supwsd

        self.conf(model_path)

        test.callback(paths["suptag"], paths["supkey"])
        with open(paths["unikey"]) as goldkey, open(
            pjoin(model_path, "scores/plain.result"), "rb"
        ) as supwsd_result_fp, open(guess_fn, "w") as guess_fp:
            proc_supwsd(goldkey, supwsd_result_fp, guess_fp)


class Elmo(SupExp):
    def __init__(self, layer):
        self.layer = layer

        super().__init__(
            ["Supervised", "ELMo-NN"],
            f"elmo_nn.{layer}",
            "ELMO-NN ({})".format(layer),
            None,
            {"layer": layer},
        )

    def train(self, paths, model_path):
        from wsdeval.systems.elmo import train

        with open(paths["sup"], "rb") as inf, open(paths["supkey"], "r") as keyin:
            train.callback(inf, keyin, model_path, self.layer)

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.elmo import test

        with open(paths["sup"], "rb") as inf, open(guess_fn, "w") as keyout:
            test.callback(model_path, inf, keyout, self.layer)


class Bert(Exp):
    def __init__(self, layer):
        self.layer = layer

        super().__init__(
            ["Supervised", "BERT-NN"],
            f"bert_nn.{layer}",
            "BERT-NN ({})".format(layer),
            None,
            {"layer": layer},
        )


class SesameAllExpGroup(ExpGroup):
    def get_paths(self, path_info, path, opt_dict):
        model_paths = []
        included = []

        for exp in self.exps:
            _, _, model_path, _ = exp.get_paths(path_info)
            if self.exp_included(exp, path, opt_dict):
                included.append(True)
            else:
                model_path = "/dev/null"
                included.append(False)
            model_paths.append(model_path)

        paths = get_partition_paths(path_info.corpus, "corpus")
        return paths, model_paths, included

    def get_eval_paths(self, path_info, path, opt_dict):
        guess_paths = []
        keyouts = []
        golds = []
        for exp in self.exps:
            _, guess_path, _, gold = exp.get_paths(path_info)
            if not self.exp_included(exp, path, opt_dict):
                guess_path = "/dev/null"
            guess_paths.append(guess_path)
            keyouts.append(open(guess_path, "w"))
            golds.append(gold)
        return guess_paths, keyouts, golds

    def train_all(self, path_info, path, opt_dict):
        if not self.group_included(path, opt_dict):
            return

        paths, model_paths, included = self.get_paths(path_info, path, opt_dict)

        print(f"Training {self.NAME} all")

        try:
            with open(paths["sup"], "rb") as inf, open(paths["supkey"], "r") as keyin:
                self.train_all_impl.callback(inf, keyin, model_paths)
        except Exception:
            traceback.print_exc()
            return

    def run_all(self, path_info, path, opt_dict):
        if not self.group_included(path, opt_dict):
            return

        paths, model_paths, included = self.get_paths(path_info, path, opt_dict)
        guess_paths, keyouts, golds = self.get_eval_paths(path_info, path, opt_dict)

        print(f"Running {self.NAME} all")
        try:
            with open(paths["sup"], "rb") as inf:
                self.test_all_impl.callback(inf, zip(model_paths, keyouts))
        except Exception:
            traceback.print_exc()
            return

        for keyout in keyouts:
            keyout.close()


class ElmoAllExpGroup(SesameAllExpGroup):
    from wsdeval.systems.sesame import train_elmo_all, test_elmo_all

    train_all_impl = staticmethod(train_elmo_all)
    test_all_impl = staticmethod(test_elmo_all)

    NAME = "ELMo"
    LAYERS = [-1, 0, 1, 2]

    def __init__(self):
        super().__init__([Elmo(layer) for layer in self.LAYERS])


class BertAllExpGroup(SesameAllExpGroup):
    from wsdeval.systems.sesame import train_bert_all, test_bert_all

    train_all_impl = staticmethod(train_bert_all)
    test_all_impl = staticmethod(test_bert_all)

    NAME = "BERT"
    LAYERS = list(range(12))

    def __init__(self):
        super().__init__([Bert(layer) for layer in self.LAYERS])


def baseline(*args):
    def run(paths, guess_fn):
        all_args = ["baselines.py"] + list(args) + [paths["unified"], guess_fn]
        os.chdir(SYSTEMS_DIR)
        python(*all_args)

    return run


def lesk(variety, *args):
    def run(paths, guess_fn):
        all_args = ["lesk.py", variety] + list(args) + [paths["suptag"], guess_fn]
        os.chdir(SYSTEMS_DIR)
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
            paths["unified"],
            guess_fn,
            variant,
            "support/ukb/wn30/wn30g.bin",
            dict_fn,
            extract_extra,
        )

    return run


class Ctx2Vec(SupExp):
    def __init__(self):
        super().__init__(
            ["Supervised", "Context2Vec"], "ctx2vec", "Context2Vec", None, {}
        )

    def train(self, paths, model_path):
        from wsdeval.systems.ctx2vec import train

        train.callback(paths["sup"], paths["sup3key"], model_path)

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.ctx2vec import test

        test.callback(model_path, paths["sup"], paths["sup3key"], guess_fn)


class AweNn(SupExp):
    def __init__(self, vec, mean):
        self.vec = vec
        self.mean = mean
        super().__init__(
            ["Supervised", "AWE-NN"],
            mk_nick("awe_nn", vec, mean),
            "AWE-NN ({}, {})".format(vec, MEAN_DISPS[mean]),
            None,
            {"vec": vec, "mean": mean},
        )

    def train(self, paths, model_path):
        from wsdeval.systems.nn import train

        with open(paths["suptag"], "rb") as inf, open(paths["supkey"], "r") as keyin:
            train.callback(self.vec, self.mean, inf, keyin, model_path)

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.nn import test

        with open(paths["suptag"], "rb") as inf, open(guess_fn, "w") as keyout:
            test.callback(self.vec, self.mean, model_path, inf, keyout)


def lesk_pp(mean, do_expand, exclude_cand, score_by):
    def inner(paths, guess_fn):
        args = [
            "lesk_pp.py",
            mean,
            paths["unified"],
            guess_fn,
            "--include-wfs",
            "--score-by",
            score_by,
        ]
        if do_expand:
            args.append("--expand")
        if exclude_cand:
            args.append("--exclude-cand")
        os.chdir(SYSTEMS_DIR)
        python(*args)

    return inner
