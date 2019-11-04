from dataclasses import dataclass
from os.path import join as pjoin
from stiff.eval import get_partition_paths
from typing import Optional
from expcomb.models import ExpGroup as ExpGroupBase


@dataclass(frozen=True)
class ExpPathInfo:
    corpus: str
    guess: Optional[str] = None
    guess_full: Optional[str] = None
    models: Optional[str] = None
    model_full: Optional[str] = None

    def get_paths(self, iden, exp):
        paths = get_partition_paths(self.corpus, "corpus")
        guess_path = (
            self.guess_full
            if self.guess_full
            else (self.guess.replace("__NICK__", exp.nick) if self.guess else None)
        )
        model_path = (
            self.model_full
            if self.model_full
            else (pjoin(self.models, exp.nick) if self.models else self.models)
        )
        return paths, guess_path, model_path, self.corpus


class ExpGroup(ExpGroupBase):
    group_attrs = ("sup", "gpu", "eng_sup")
    sup = False
    eng_sup = False
    gpu = False


class SupExpGroup(ExpGroup):
    sup = True


class SupGpuExpGroup(SupExpGroup):
    gpu = True


class EngSupExpGroup(ExpGroup):
    eng_sup = True
