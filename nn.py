import click
import pickle

from finntk.emb.base import BothVectorSpaceAdapter, MonoVectorSpaceAdapter
from finntk.emb.concat import ft_nb_w2v_space
from finntk.emb.fasttext import multispace as fasttext_multispace
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.emb.utils import apply_vec
from finntk.emb.word2vec import space as word2vec_space
from means import ALL_MEANS
from sup_corpus import next_key
from vec_nn_utils import mk_training_examples, train_vec_nn, test_vec_nn


def get_vec_space(vec):
    if vec == "numberbatch":
        return BothVectorSpaceAdapter(MonoVectorSpaceAdapter(numberbatch_multispace, "fi"))
    elif vec == "fasttext":
        return BothVectorSpaceAdapter(MonoVectorSpaceAdapter(fasttext_multispace, "fi"))
    elif vec == "word2vec":
        return BothVectorSpaceAdapter(word2vec_space)
    elif vec == "triple":
        return ft_nb_w2v_space
    else:
        assert False


@click.group()
def nn():
    pass


def iter_inst_ctxs(inf, aggf, space):
    from sup_corpus import iter_instances, norm_wf_lemma_of_tokens
    for inst_id, item_pos, (be, he, af) in iter_instances(inf):
        ctx = norm_wf_lemma_of_tokens(be + af)
        ctx_vec = apply_vec(aggf, space, ctx, "fi") if ctx else None
        yield inst_id, item_pos, ctx_vec


@nn.command("train")
@click.argument("vec")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.File("wb"))
def train(vec, mean, inf, keyin, model):
    """
    Train nearest neighbour classifier.
    """
    space = get_vec_space(vec)
    aggf = ALL_MEANS[mean]
    classifier = train_vec_nn(mk_training_examples(iter_inst_ctxs(inf, aggf, space), keyin))

    pickle.dump(classifier, model)


@nn.command("test")
@click.argument("vec")
@click.argument("mean")
@click.argument("model", type=click.File("rb"))
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
def test(vec, mean, model, inf, keyout):
    """
    Test nearest neighbour classifier.
    """
    space = get_vec_space(vec)
    aggf = ALL_MEANS[mean]
    classifier = pickle.load(model)
    test_vec_nn(classifier, iter_inst_ctxs(inf, aggf, space), keyout)


if __name__ == '__main__':
    nn()
