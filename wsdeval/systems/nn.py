import os
import click
from itertools import starmap

from finntk.emb.base import BothVectorSpaceAdapter, MonoVectorSpaceAdapter
from finntk.emb.concat import ft_nb_multispace, ft_nb_w2v_space
from finntk.emb.fasttext import multispace as fasttext_multispace
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.emb.utils import apply_vec
from finntk.emb.word2vec import space as word2vec_space
from wsdeval.nn.vec_nn_common import mk_training_examples, normalize
from wsdeval.tools.means import EXPANSION_FACTOR, get_mean


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


def iter_inst_ctxs(inf, aggf, space, synsets=False, do_new=True):
    import numpy
    from wsdeval.formats.sup_corpus import (
        iter_instances_grouped,
        norm_wf_lemma_of_tokens,
    )
    from wsdeval.nn.new import TYPE

    def calc_vec(inst_id, texts):
        (be, he, af) = texts
        ctx = norm_wf_lemma_of_tokens(be + af)
        ctx_vec = None
        if ctx:
            ctx_vec = apply_vec(aggf, space, ctx, "fi", dtype=TYPE if do_new else None)
            if ctx_vec is not None:
                if (not numpy.isfinite(ctx_vec).all()) or (not numpy.any(ctx_vec)):
                    ctx_vec = None
                elif do_new:
                    ctx_vec = normalize(ctx_vec)
        return inst_id, ctx_vec

    for group_raw, cnt, it in iter_instances_grouped(inf, synsets=synsets):
        if synsets:
            group = group_raw
        else:
            group = ".".join(group_raw)
        yield group, cnt, starmap(calc_vec, it)


@nn.command("train")
@click.argument("vec")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.Path())
@click.option("--synsets/--words")
def train(vec, mean, inf, keyin, model, synsets):
    """
    Train nearest neighbour classifier.
    """
    space = get_vec_space(vec)
    aggf = get_mean(mean, vec)
    use_old = os.environ.get("USE_OLD_NN")
    training_examples = mk_training_examples(
        iter_inst_ctxs(inf, aggf, space, synsets=synsets, do_new=not use_old), keyin
    )
    if use_old:
        assert not synsets
        from wsdeval.nn.old import WordExpertManager
        from wsdeval.nn.vec_nn_old import train_vec_nn

        train_vec_nn(WordExpertManager(model, "w"), training_examples)
    else:
        from wsdeval.nn.new import GroupedVecExactNN
        from wsdeval.nn.vec_nn_new import train_word_experts, train_synset_examples

        if synsets:
            manager = GroupedVecExactNN(
                model, space.dim * EXPANSION_FACTOR.get(mean, 1), "wd"
            )
            train_synset_examples(manager, training_examples)
        else:
            manager = GroupedVecExactNN(
                model, space.dim * EXPANSION_FACTOR.get(mean, 1), "wd", value_bytes=10
            )
            train_word_experts(manager, training_examples)


@nn.command("test")
@click.argument("vec")
@click.argument("mean")
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--use-freq/--no-use-freq")
@click.option("--synsets/--words")
def test(vec, mean, model, inf, keyout, use_freq, synsets):
    """
    Test nearest neighbour classifier.
    """
    space = get_vec_space(vec)
    aggf = get_mean(mean, vec)
    use_old = os.environ.get("USE_OLD_NN")
    inst_ctxs = iter_inst_ctxs(inf, aggf, space, do_new=not use_old)
    if use_old:
        assert not synsets
        from wsdeval.nn.old import WordExpertManager
        from wsdeval.nn.vec_nn_old import test_vec_nn

        test_vec_nn(
            WordExpertManager(model), inst_ctxs, keyout, allow_most_freq=use_freq
        )
    else:
        from wsdeval.nn.new import GroupedVecExactNN
        from wsdeval.nn.vec_nn_new import test_synset_experts, test_word_experts

        if synsets:
            manager = GroupedVecExactNN(
                model, space.dim * EXPANSION_FACTOR.get(mean, 1), "rd"
            )
            test_synset_experts(manager, inst_ctxs, keyout, allow_most_freq=use_freq)
        else:
            manager = GroupedVecExactNN(
                model, space.dim * EXPANSION_FACTOR.get(mean, 1), "rd", value_bytes=10
            )
            test_word_experts(manager, inst_ctxs, keyout, allow_most_freq=use_freq)


if __name__ == "__main__":
    nn()
