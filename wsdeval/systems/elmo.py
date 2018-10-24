import os
import click
from wsdeval.vec_nn_utils import mk_training_examples, train_vec_nn, test_vec_nn
from finntk.wsd.nn import WordExpertManager


def get_batch_size():
    return int(os.environ.get("BATCH_SIZE", "32"))


@click.group()
def elmo():
    pass


def iter_inst_vecs(inf, output_layer, batch_size=None):
    from wsdeval.sup_corpus import iter_instances
    from finntk.emb.elmo import vecs
    from finntk.vendor.elmo import embed_sentences

    if batch_size is None:
        batch_size = get_batch_size()

    model = vecs.get()
    iter = iter_instances(inf)
    while 1:
        infos = []
        sents = []
        is_end = True
        for idx, (inst_id, item_pos, (be, he, af)) in enumerate(iter):
            sents.append(be + he + af)
            start_idx = len(be)
            end_idx = len(be) + len(he)
            infos.append((inst_id, item_pos, start_idx, end_idx))
            if (idx + 1) == batch_size:
                is_end = False
                break
        embs = embed_sentences(model, sents, output_layer)
        for (inst_id, item_pos, start_idx, end_idx), emb in zip(infos, embs):
            vec = emb[start_idx:end_idx].mean(axis=0)
            yield inst_id, item_pos, vec
        if is_end:
            break


@elmo.command("train")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.Path())
@click.argument("output_layer", type=int)
def train(inf, keyin, model, output_layer):
    """
    Train nearest neighbour classifier with ELMo.
    """
    train_vec_nn(
        WordExpertManager(model, "w"),
        mk_training_examples(iter_inst_vecs(inf, output_layer), keyin),
    )


@elmo.command("test")
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.argument("output_layer", type=int)
def test(model, inf, keyout, output_layer):
    """
    Test nearest neighbour classifier.
    """
    test_vec_nn(WordExpertManager(model), iter_inst_vecs(inf, output_layer), keyout)
