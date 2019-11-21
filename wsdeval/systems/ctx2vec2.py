import click
from wsdeval.tools.ctx_embedder import ctx2vec2_embedder
from vecstorenn import VecStorage
from wsdeval.nn.vec_nn_new import train_word_experts, test_word_experts
from wsdeval.nn.vec_nn_common import mk_training_examples

CTX2VEC2_SIZE = 600


@click.group()
def ctx2vec2():
    pass


@ctx2vec2.command()
@click.argument("inf", type=click.Path())
@click.argument("keyin", type=click.Path())
@click.argument("model", type=click.Path())
def train(inf, keyin, model):
    inst_it = ctx2vec2_embedder.iter_inst_vecs_grouped(inf)
    training_examples = mk_training_examples(inst_it, keyin)

    manager = VecStorage(model, CTX2VEC2_SIZE, "wd", value_bytes=10)
    train_word_experts(manager, training_examples)


@ctx2vec2.command()
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.Path())
@click.argument("keyout", type=click.Path())
@click.option("--use-freq/--no-use-freq")
def test(model, inf, keyout, use_freq):
    inst_it = ctx2vec2_embedder.iter_inst_vecs_grouped(inf)

    manager = VecStorage(model, CTX2VEC2_SIZE, "rd", value_bytes=10)
    test_word_experts(manager, inst_it, keyout, allow_most_freq=use_freq)


if __name__ == "__main__":
    ctx2vec2()
