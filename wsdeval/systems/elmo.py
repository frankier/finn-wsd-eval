import os
import click
from wsdeval.vec_nn_utils import (
    mk_training_examples,
    train_vec_nn,
    test_vec_nn,
    train_many_vec_nn,
    test_many_vec_nn,
)
from finntk.wsd.nn import WordExpertManager

LAYERS = [-1, 0, 1, 2]


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
            if output_layer == -2:
                vec = emb[:, start_idx:end_idx]
            else:
                vec = emb[start_idx:end_idx]
            vec.shape = vec.shape[:-2] + (vec.shape[-2] * vec.shape[-1],)
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
    Train nearest neighbour classifier with ELMo for all layers.
    """

    inst_it = iter_inst_vecs(inf, output_layer)
    training_examples = mk_training_examples(inst_it, keyin)
    train_vec_nn(WordExpertManager(model, "w"), training_examples)


@elmo.command("train-all")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("models", type=click.Path(), nargs=-1)
def train_all(inf, keyin, models):
    """
    Train nearest neighbour classifier with ELMo.
    """
    inst_it = iter_inst_vecs(inf, -2)
    training_examples = mk_training_examples(inst_it, keyin)
    expert_managers = []
    for model in models:
        expert_managers.append(WordExpertManager(model, "w"))
    train_many_vec_nn(expert_managers, training_examples)


@elmo.command("test")
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.argument("output_layer", type=int)
def test(model, inf, keyout, output_layer):
    """
    Test nearest neighbour classifier.
    """
    inst_it = iter_inst_vecs(inf, output_layer)
    test_vec_nn(WordExpertManager(model, "w"), inst_it, keyout)


@elmo.command("test-all")
@click.argument("inf", type=click.File("rb"))
@click.argument("model_keyouts", type=(click.Path(), click.File("w")))
def test_all(inf, model_keyouts):
    """
    Test nearest neighbour classifier for all layers.
    """
    inst_it = iter_inst_vecs(inf, -2)
    expert_managers = []
    keyouts = []
    for model, keyout in model_keyouts:
        expert_managers.append(WordExpertManager(model, "r"))
        keyouts.append(keyout)
    test_many_vec_nn(expert_managers, inst_it, keyouts)
