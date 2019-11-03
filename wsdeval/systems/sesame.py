import click
import logging
from wsdeval.vec_nn_common import mk_training_examples
from wsdeval.nn.vec_nn_old import (
    train_vec_nn,
    test_vec_nn,
    train_many_vec_nn,
    test_many_vec_nn,
)
from finntk.wsd.nn import WordExpertManager

LAYERS = [-1, 0, 1, 2]
logger = logging.getLogger(__name__)


@click.group()
def elmo():
    pass


def train_sesame_all(inst_it, keyin, models):
    training_examples = mk_training_examples(inst_it, keyin)
    expert_managers = []
    for model in models:
        logger.debug(f"train_sesame_all: model: {model}")
        if model == "/dev/null":
            expert_managers.append(None)
        else:
            expert_managers.append(WordExpertManager(model, "w"))
    train_many_vec_nn(expert_managers, training_examples)


def test_sesame_all(inst_it, model_keyouts):
    expert_managers = []
    keyouts = []
    for model, keyout in model_keyouts:
        logger.debug(f"test_sesame_all: model: {model}, keyout: {keyout}")
        if model == "/dev/null":
            expert_managers.append(None)
            keyouts.append(None)
        else:
            expert_managers.append(WordExpertManager(model, "r"))
            keyouts.append(keyout)
    test_many_vec_nn(expert_managers, inst_it, keyouts)


@elmo.command("train-elmo")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.Path())
@click.argument("output_layer", type=int)
def train_elmo(inf, keyin, model, output_layer):
    """
    Train nearest neighbour classifier with ELMo for one layer.
    """
    from wsdeval.tools.ctx_embedder import elmo_embedder

    inst_it = elmo_embedder.iter_inst_vecs_grouped(inf, output_layer=output_layer)
    training_examples = mk_training_examples(inst_it, keyin)
    train_vec_nn(WordExpertManager(model, "w"), training_examples)


@elmo.command("train-elmo-all")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("models", type=click.Path(), nargs=-1)
def train_elmo_all(inf, keyin, models):
    """
    Train nearest neighbour classifier with ELMo for all layers.
    """
    from wsdeval.tools.ctx_embedder import elmo_embedder

    inst_it = elmo_embedder.iter_inst_vecs_grouped(inf, output_layer=-2)
    train_sesame_all(inst_it, keyin, models)


@elmo.command("train-bert-all")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("models", type=click.Path(), nargs=-1)
def train_bert_all(inf, keyin, models):
    """
    Train nearest neighbour classifier with BERT for all layers.
    """
    from wsdeval.tools.ctx_embedder import bert_embedder

    inst_it = bert_embedder.iter_inst_vecs_grouped(inf)
    train_sesame_all(inst_it, keyin, models)


@elmo.command("test-elmo")
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.argument("output_layer", type=int)
def test_elmo(model, inf, keyout, output_layer):
    """
    Test nearest neighbour ELMo classifier for one layer.
    """
    from wsdeval.tools.ctx_embedder import elmo_embedder

    inst_it = elmo_embedder.iter_inst_vecs_grouped(inf, output_layer=output_layer)
    test_vec_nn(WordExpertManager(model, "w"), inst_it, keyout)


@elmo.command("test-elmo-all")
@click.argument("inf", type=click.File("rb"))
@click.argument("model_keyouts", type=(click.Path(), click.File("w")))
def test_elmo_all(inf, model_keyouts):
    """
    Test nearest neighbour ELMo classifier for all layers.
    """
    from wsdeval.tools.ctx_embedder import elmo_embedder

    inst_it = elmo_embedder.iter_inst_vecs_grouped(inf, output_layer=-2)
    test_sesame_all(inst_it, model_keyouts)


@elmo.command("test-bert-all")
@click.argument("inf", type=click.File("rb"))
@click.argument("model_keyouts", type=(click.Path(), click.File("w")))
def test_bert_all(inf, model_keyouts):
    """
    Test nearest neighbour BERT classifier for all layers.
    """
    from wsdeval.tools.ctx_embedder import bert_embedder

    inst_it = bert_embedder.iter_inst_vecs_grouped(inf)
    test_sesame_all(inst_it, model_keyouts)
