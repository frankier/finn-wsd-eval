from dataclasses import dataclass
from os.path import join as pjoin
from stiff.eval import get_partition_paths
from typing import Optional


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
            else (pjoin(self.guess, iden) if self.guess else None)
        )
        model_path = (
            self.model_full
            if self.model_full
            else (pjoin(self.models, iden) if self.models else self.models)
        )
        return paths, guess_path, model_path, self.corpus
