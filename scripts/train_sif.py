import click
import pickle
import numpy as np
from os import makedirs
from os.path import join as pjoin

from stiff.eval import get_eval_paths
from stiff.sup_corpus import iter_instances, norm_wf_lemma_of_tokens
from finntk.emb.utils import apply_vec, compute_pc
from stiff.utils.xml import iter_blocks
from finntk.emb.base import MonoVectorSpaceAdapter
from finntk.emb.utils import pre_sif_mean


def get_vec_space(vec):
    from nn import get_vec_space
    from finntk.emb.concat import ft_nb_multispace

    if vec == "double":
        return MonoVectorSpaceAdapter(ft_nb_multispace, "fi")
    else:
        return get_vec_space(vec)


def train_one_sif(inf, vec):
    space = get_vec_space(vec)
    instances = 0
    for _ in iter_blocks("instance")(inf):
        instances += 1
    inf.seek(0)
    mat = np.zeros((instances, space.dim))
    for idx, (_, _, (be, _, af)) in enumerate(iter_instances(inf)):
        ctx = norm_wf_lemma_of_tokens(be + af)
        if not ctx:
            continue
        vec = apply_vec(pre_sif_mean, space, ctx, "fi")
        if vec is None:
            continue
        mat[idx] = vec
    return compute_pc(mat)


def train_one_sif_outer(paths, vec, out_path):
    pc = train_one_sif(open(paths["train"]["suptag"], "rb"), vec)
    pickle.dump(pc, open(sif_filename(out_path, vec), "wb"))


def sif_filename(path, vec):
    makedirs(path, exist_ok=True)
    return pjoin(path, f"{vec}.pkl")


@click.command()
@click.argument("corpus", type=click.Path())
@click.argument("out_path", type=click.Path())
@click.argument("vec", required=False)
def train_sif(corpus, out_path, vec=None):
    root, paths = get_eval_paths(corpus)
    if vec is None:
        for vec in ["numberbatch", "fasttext", "word2vec", "triple", "double"]:
            train_one_sif_outer(paths, vec, out_path)
    else:
        train_one_sif_outer(paths, vec, out_path)


def load_sif(path, vec):
    return pickle.load(open(sif_filename(path, vec)))


if __name__ == "__main__":
    train_sif()
