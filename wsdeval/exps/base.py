import traceback
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from tinyrecord import transaction
from .utils import score
from stiff.eval import get_partition_paths
from wsdeval.exps.utils import mk_iden, mk_guess_path, mk_model_path


@dataclass(frozen=True)
class ExpPathInfo:
    corpus: str
    guess: str
    models: str


@dataclass(frozen=True)
class Exp:
    category: str
    subcat: str
    nick: str
    disp: str
    run_func: Optional[Callable[[str, str, str], None]] = None
    opts: Dict[str, any] = field(default_factory=dict)
    lex_group: bool = False

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

    def get_iden(self, path_info):
        return mk_iden(path_info, self)

    def get_paths(self, path_info):
        paths = get_partition_paths(path_info.corpus, "corpus")
        iden = self.get_iden(path_info)
        guess_path = mk_guess_path(path_info, iden)
        model_path = mk_model_path(path_info, iden)
        if self.lex_group:
            gold = paths["supkey"]
        else:
            gold = paths["unikey"]
        return paths, guess_path, model_path, gold

    def run_dispatch(self, paths, guess_path, model_path):
        return self.run(paths, guess_path)

    def run_and_score(self, db, path_info, score=True):
        paths, guess_path, model_path, gold = self.get_paths(path_info)
        try:
            self.run_dispatch(paths, guess_path, model_path)
        except Exception:
            traceback.print_exc()
            return
        if score:
            return self.proc_score(db, path_info, gold, guess_path)
        else:
            return guess_path

    def proc_score(self, db, path_info, gold, guess):
        measures = score(gold, guess)

        result = self.info()
        result.update(measures)
        result["corpus"] = path_info.corpus
        result["time"] = time.time()

        with transaction(db) as tr:
            tr.insert(result)
        return measures


class SupExp(Exp):
    def train_model(self, path_info):
        paths, _, model_path, _ = self.get_paths(path_info)
        self.train(paths, model_path)

    def run_dispatch(self, paths, guess_path, model_path):
        return self.run(paths, guess_path, model_path)


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

    def group_included(self, filter_l1, filter_l2, opt_dict):
        return any(
            (
                self.exp_included(exp, filter_l1, filter_l2, opt_dict)
                for exp in self.exps
            )
        )

    def train_all(self, path_info, filter_l1, filter_l2, opt_dict):
        for exp in self.filter_exps(filter_l1, filter_l2, opt_dict):
            if isinstance(exp, SupExp):
                print("Training", exp)
                exp.train_model(path_info)

    def run_all(self, db, path_info, filter_l1, filter_l2, opt_dict, score=True):
        for exp in self.filter_exps(filter_l1, filter_l2, opt_dict):
            print("Running", exp)
            measures = exp.run_and_score(db, path_info, score=score)
            print("Got", measures)
