import click
from itertools import starmap

from finntk.emb.base import BothVectorSpaceAdapter, MonoVectorSpaceAdapter
from finntk.emb.concat import ft_nb_multispace, ft_nb_w2v_space
from finntk.emb.fasttext import multispace as fasttext_multispace
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.emb.utils import apply_vec
from finntk.emb.word2vec import space as word2vec_space
from wsdeval.tools.means import get_mean
from wsdeval.tools.vec_nn import mk_training_examples, train_vec_nn, test_vec_nn
from finntk.wsd.nn import WordExpertManager


def get_vec_space(vec):
    if vec == "numberbatch":
        return BothVectorSpaceAdapter(
            MonoVectorSpaceAdapter(numberbatch_multispace, "fi")
        )
    elif vec == "fasttext":
        return BothVectorSpaceAdapter(MonoVectorSpaceAdapter(fasttext_multispace, "fi"))
    elif vec == "word2vec":
        return BothVectorSpaceAdapter(word2vec_space)
    elif vec == "double":
        return BothVectorSpaceAdapter(MonoVectorSpaceAdapter(ft_nb_multispace, "fi"))
    elif vec == "triple":
        return ft_nb_w2v_space
    else:
        assert False


@click.group()
def nn():
    pass


def iter_inst_ctxs(inf, aggf, space):
    import numpy
    from wsdeval.formats.sup_corpus import (
        iter_instances_grouped,
        norm_wf_lemma_of_tokens,
    )

    def calc_vec(inst_id, texts):
        (be, he, af) = texts
        ctx = norm_wf_lemma_of_tokens(be + af)
        ctx_vec = None
        if ctx:
            ctx_vec = apply_vec(aggf, space, ctx, "fi")
            if (ctx_vec is not None) and (
                (not numpy.isfinite(ctx_vec).all()) or (not numpy.any(ctx_vec))
            ):
                ctx_vec = None
        return inst_id, ctx_vec

    for item_pos, cnt, it in iter_instances_grouped(inf):
        yield ".".join(item_pos), cnt, starmap(calc_vec, it)


@nn.command("train")
@click.argument("vec")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.Path())
def train(vec, mean, inf, keyin, model):
    """
    Train nearest neighbour classifier.
    """
    space = get_vec_space(vec)
    aggf = get_mean(mean, vec)
    train_vec_nn(
        WordExpertManager(model, "w"),
        mk_training_examples(iter_inst_ctxs(inf, aggf, space), keyin),
    )


@nn.command("test")
@click.argument("vec")
@click.argument("mean")
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
def test(vec, mean, model, inf, keyout):
    """
    Test nearest neighbour classifier.
    """
    space = get_vec_space(vec)
    aggf = get_mean(mean, vec)
    test_vec_nn(WordExpertManager(model), iter_inst_ctxs(inf, aggf, space), keyout)


if __name__ == "__main__":
    nn()
