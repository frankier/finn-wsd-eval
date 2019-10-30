import os
from plumbum.cmd import python
from plumbum import local
from expcomb.models import Exp, SupExp
from expcomb.filter import SimpleFilter
from .utils import mk_nick
from stiff.eval import get_partition_paths
import traceback
from datetime import datetime
from os import makedirs, symlink
from os.path import abspath, dirname, exists, join as pjoin
import shutil
from wsdeval.tools.means import MEAN_DISPS
from .base import SupGpuExpGroup
from hashlib import sha256


def relpath(rel):
    return pjoin(dirname(abspath(__file__)), rel)


SYSTEMS_DIR = relpath("../systems")
SUPPORT_DIR = relpath("../../support")
orig_cwd = os.getcwd()


def cwd_relpath(rel):
    return pjoin(orig_cwd, rel)


def setup_paths():
    local.env["WSDEVAL_SUPPORT"] = SUPPORT_DIR
    os.chdir(SYSTEMS_DIR)


def timestr():
    return datetime.now().isoformat()


def hash_paths(paths):
    return (sha256(path.encode("utf-8")).hexdigest() for path in paths)


def ensure_symlink(src_par, dst_par, sub):
    src = pjoin(src_par, sub)
    dst = pjoin(dst_par, sub)
    if not exists(dst):
        symlink(src, dst)


class SupWSD(SupExp):
    def __init__(self, vec, sur_words):
        vec_path = "support/emb/{}.txt".format(vec) if vec is not None else ""
        no_sur = "-s" if not sur_words else ""
        disp = f"SupWSD\\textsubscript{{{vec}{no_sur}}}"

        self.vec_path = vec_path
        self.use_vec = vec is not None
        self.sur_words = sur_words
        nick = mk_nick("supwsd", vec, (sur_words, "sur"))
        super().__init__(
            ["Supervised", "SupWSD"],
            nick,
            disp,
            None,
            {"vec": vec, "sur_words": sur_words},
        )

    def get_conf_dir(self, type, *paths):
        return abspath(
            pjoin("systems", "supwsd_confs", self.nick, type, *hash_paths(paths))
        )

    def get_work_dir(self, model_path, type, *paths):
        work_dir = abspath(pjoin(model_path, type, *hash_paths(paths)))
        makedirs(work_dir, exist_ok=True)
        return work_dir

    def conf(self, work_dir, conf_dir):
        from wsdeval.systems.supwsd import conf

        conf.callback(
            work_dir=work_dir,
            vec_path=abspath(self.vec_path),
            dest=conf_dir,
            use_vec=self.use_vec,
            use_surrounding_words=self.sur_words,
        )

    def train(self, paths, model_path):
        from wsdeval.systems.supwsd import train

        work_dir = self.get_work_dir(model_path, "train")
        conf_dir = self.get_conf_dir("train", model_path)
        self.conf(work_dir, conf_dir)

        if exists(model_path):
            shutil.move(model_path, "{}.{}".format(model_path, timestr()))
        makedirs(model_path, exist_ok=True)
        train.callback(paths["suptag"], paths["supkey"], conf_dir)

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.supwsd import test
        from wsdeval.formats.supwsd import proc_supwsd

        train_work_dir = self.get_work_dir(model_path, "train")
        work_dir = self.get_work_dir(model_path, "test", guess_fn)
        conf_dir = self.get_conf_dir("test", model_path, guess_fn)
        self.conf(work_dir, conf_dir)
        ensure_symlink(train_work_dir, work_dir, "models")
        ensure_symlink(train_work_dir, work_dir, "stat")

        test.callback(paths["suptag"], paths["supkey"], conf_dir)
        with open(paths["supkey"]) as goldkey, open(
            pjoin(work_dir, "scores/plain.result"), "rb"
        ) as supwsd_result_fp, open(cwd_relpath(guess_fn), "w") as guess_fp:
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

        with open(paths["sup"], "rb") as inf, open(
            cwd_relpath(guess_fn), "w"
        ) as keyout:
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


class SesameAllExpGroup(SupGpuExpGroup):
    group_at_once = True

    def get_paths(self, path_info, filter: SimpleFilter):
        model_paths = []
        included = []

        for exp in self.exps:
            _, _, model_path, _ = exp.get_paths_from_path_info(path_info)
            if self.exp_included(exp, filter):
                included.append(True)
            else:
                model_path = "/dev/null"
                included.append(False)
            model_paths.append(model_path)

        paths = get_partition_paths(path_info.corpus, "corpus")
        return paths, model_paths, included

    def get_eval_paths(self, path_info, filter: SimpleFilter):
        guess_paths = []
        keyouts = []
        golds = []
        for exp in self.exps:
            _, guess_path, _, gold = exp.get_paths_from_path_info(path_info)
            if not self.exp_included(exp, filter):
                guess_path = "/dev/null"
            guess_paths.append(guess_path)
            keyouts.append(open(guess_path, "w"))
            golds.append(gold)
        return guess_paths, keyouts, golds

    def train_all(self, path_info, filter: SimpleFilter):
        if not self.group_included(filter):
            return

        paths, model_paths, included = self.get_paths(path_info, filter)

        print(f"Training {self.NAME} all")

        try:
            with open(paths["sup"], "rb") as inf, open(paths["supkey"], "r") as keyin:
                self.train_all_impl.callback(inf, keyin, model_paths)
        except Exception:
            traceback.print_exc()
            return

    def run_all(self, path_info, filter: SimpleFilter):
        if not self.group_included(filter):
            return

        paths, model_paths, included = self.get_paths(path_info, filter)
        guess_paths, keyouts, golds = self.get_eval_paths(path_info, filter)

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
        all_args = (
            ["baselines.py"] + list(args) + [paths["unified"], cwd_relpath(guess_fn)]
        )
        setup_paths()
        python(*all_args)

    return run


def lesk(variety, *args):
    def run(paths, guess_fn):
        all_args = (
            ["lesk.py", variety] + list(args) + [paths["suptag"], cwd_relpath(guess_fn)]
        )
        setup_paths()
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
            cwd_relpath(guess_fn),
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

        abs_model_path = cwd_relpath(model_path)
        makedirs(abs_model_path, exist_ok=True)
        full_model_path = pjoin(abs_model_path, "model")

        train.callback(
            cwd_relpath(paths["sup"]), cwd_relpath(paths["sup3key"]), full_model_path
        )

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.ctx2vec import test

        abs_model_path = cwd_relpath(model_path)
        full_model_path = pjoin(abs_model_path, "model")

        test.callback(
            full_model_path,
            cwd_relpath(paths["sup"]),
            cwd_relpath(paths["sup3key"]),
            cwd_relpath(guess_fn),
        )


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

        with open(paths["suptag"], "rb") as inf, open(
            cwd_relpath(guess_fn), "w"
        ) as keyout:
            test.callback(self.vec, self.mean, model_path, inf, keyout, use_freq=True)


def lesk_pp(mean, do_expand, exclude_cand, score_by):
    def inner(paths, guess_fn):
        args = [
            "lesk_pp.py",
            mean,
            paths["unified"],
            cwd_relpath(guess_fn),
            "--include-wfs",
            "--score-by",
            score_by,
        ]
        if do_expand:
            args.append("--expand")
        if exclude_cand:
            args.append("--exclude-cand")
        setup_paths()
        python(*args)

    return inner


class Floor(Exp):
    def __init__(self):
        super().__init__(["Limits", "Floor"], "floor", "Floor")

    def run(self, paths, guess_fn):
        from wsdeval.systems.limits import floor

        with open(paths["suptag"], "rb") as inf, open(
            cwd_relpath(guess_fn), "w"
        ) as keyout:
            floor.callback(inf, keyout)


class Ceil(SupExp):
    def __init__(self, tok):
        self.tok = tok
        if tok:
            disp_tok = "Tok"
        else:
            disp_tok = "Inst"
        super().__init__(
            ["Limits", "Ceil" + disp_tok],
            mk_nick("ceil", (tok, {False: "inst", True: "tok"})),
            "Ceil" + disp_tok,
        )

    def train(self, paths, model_path):
        from wsdeval.systems.limits import train_ceil

        with open(paths["suptag"], "rb") as inf, open(paths["supkey"], "r") as keyin:
            train_ceil.callback(inf, keyin, model_path)

    def run(self, paths, guess_fn, model_path):
        from wsdeval.systems.limits import test_ceil

        with open(paths["suptag"], "rb") as inf, open(
            paths["supkey"], "r"
        ) as keyin, open(cwd_relpath(guess_fn), "w") as keyout:
            test_ceil.callback(
                model_path, inf, keyin, keyout, "--tok" if self.tok else "--inst"
            )


class Post1stSenseCombExp(Exp):
    def __init__(self, inner_exp, x1st):
        if x1st:
            self.type = "x1st"
        else:
            self.type = "u1st"
        self.inner_exp = inner_exp
        super().__init__(
            inner_exp.path[:],
            inner_exp.nick + "." + type,
            inner_exp.disp + "+" + type,
            None,
            {"1stsensecomb": type, **inner_exp.opts},
        )

    def run(self, paths, guess_fn):
        import pickle
        from wsdeval.formats.sup_corpus import iter_keys, iter_instances_grouped

        with open(os.environ["GUESS_1ST"], "r") as key1stin:
            gold_map = dict(iter_keys(key1stin))
        with open(os.environ["INNER_GUESS"], "rb") as inkey, open(
            cwd_relpath(guess_fn), "w"
        ) as keyout:
            if self.type == "x1st":
                # Check each word to see if 1st was available when guessing: if not replace with 1st
                inst_lemmas = {}
                lemma_senses = {}
                ceil_model = os.environ["CEIL_MODEL"]
                with open(paths["suptag"], "rb") as inf:
                    for item_pos, cnt, it in iter_instances_grouped(inf):
                        item_pos_str = ".".join(item_pos)
                        with open(pjoin(ceil_model, item_pos_str), "rb") as modelf:
                            lemma_senses[item_pos_str] = pickle.load(modelf, protocol=4)
                        for inst_id, _ in it:
                            inst_lemmas[inst_id] = item_pos_str
                for k, senses in iter_keys(inkey):
                    if len(set(gold_map[k]) & set(lemma_senses[inst_lemmas[k]])):
                        # It was available: pass through
                        good_senses = senses
                    else:
                        # It wasn't: replace
                        good_senses = gold_map[k]
                    keyout.write("{} {}\n".format(k, " ".join(good_senses)))
            else:
                # Replace each "U" with GOLD
                for k, senses in iter_keys(inkey):
                    if senses != ["U"]:
                        # It was given: pass through
                        good_senses = senses
                    else:
                        # It wasn't: replace
                        good_senses = gold_map[k]
