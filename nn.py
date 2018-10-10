import click
import pickle

from finntk.emb.base import BothVectorSpaceAdapter, MonoVectorSpaceAdapter
from finntk.emb.concat import ft_nb_w2v_space
from finntk.emb.fasttext import multispace as fasttext_multispace
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.emb.utils import apply_vec
from finntk.emb.word2vec import space as word2vec_space
from means import ALL_MEANS


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
    return train_nn(
        get_vec_space(vec),
        ALL_MEANS[mean],
        inf,
        keyin,
        model,
    )


def train_nn(space, aggf, inf, keyin, model):
    from finntk.wsd.nn import WsdNn
    classifier = WsdNn()

    prev_item = None
    for inst_id, item, ctx_vec in iter_inst_ctxs(inf, aggf, space):
        key_id, synset_id = next(keyin).strip().split()
        assert inst_id == key_id
        if ctx_vec is not None:
            classifier.add_word(item, ctx_vec, synset_id)

        if prev_item is not None and item != prev_item:
            classifier.fit_word(prev_item)
        prev_item = item
    classifier.fit_word(prev_item)

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
    return test_nn(
        get_vec_space(vec),
        ALL_MEANS[mean],
        inf,
        keyout,
        model,
    )


def test_nn(space, aggf, inf, keyout, model):
    classifier = pickle.load(model)

    for inst_id, item, ctx_vec in iter_inst_ctxs(inf, aggf, space):
        prediction = None
        if ctx_vec is not None:
            try:
                prediction = classifier.predict(item, ctx_vec)
            except KeyError:
                pass
        if prediction is None:
            prediction = "U"
        keyout.write("{} {}\n".format(inst_id, prediction))


if __name__ == '__main__':
    nn()
