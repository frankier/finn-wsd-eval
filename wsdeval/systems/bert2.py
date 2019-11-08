from wsdeval.nn.vec_nn_common import mk_training_examples
from wsdeval.tools.ctx_embedder import bert2_embedder
from wsdeval.nn.new import GroupedVecExactNN
from wsdeval.nn.vec_nn_new import (
    train_word_experts,
    train_synset_examples,
    test_synset_experts,
    test_word_experts,
)

import click
import logging

logging.basicConfig(level=logging.INFO)


BERT_SIZE = 768


@click.group()
def bert2():
    pass


@bert2.command()
@click.argument("inf", type=click.Path())
@click.argument("keyin", type=click.Path())
@click.argument("model", type=click.Path())
@click.option("--synsets/--words")
def train(inf, keyin, model, synsets):
    inst_it = bert2_embedder.iter_inst_vecs_grouped(inf, synsets=synsets)
    training_examples = mk_training_examples(inst_it, keyin)

    if synsets:
        manager = GroupedVecExactNN(model, BERT_SIZE, "wd")
        train_synset_examples(manager, training_examples)
    else:
        manager = GroupedVecExactNN(model, BERT_SIZE, "wd", value_bytes=10)
        train_word_experts(manager, training_examples)


@bert2.command()
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.Path())
@click.argument("keyout", type=click.Path())
@click.option("--use-freq/--no-use-freq")
@click.option("--synsets/--words")
def test(model, inf, keyout, use_freq, synsets):
    inst_it = bert2_embedder.iter_inst_vecs_grouped(inf, synsets=False)

    if synsets:
        manager = GroupedVecExactNN(model, BERT_SIZE, "rd")
        test_synset_experts(manager, inst_it, keyout, allow_most_freq=use_freq)
    else:
        manager = GroupedVecExactNN(model, BERT_SIZE, "rd", value_bytes=10)
        test_word_experts(manager, inst_it, keyout, allow_most_freq=use_freq)


if __name__ == "__main__":
    bert2()
