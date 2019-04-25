import click
from expcomb.cmd import mk_expcomb, TinyDBParam
from wsdeval.exps.base import ExpPathInfo
from wsdeval.exps.confs import EXPERIMENTS
from wsdeval.exps.utils import score as eval_func
from os.path import join as pjoin


def extra_pk(doc):
    extra = {"test-corpus": doc["test-corpus"]}
    if "train-corpus" in doc:
        extra["train-corpus"] = doc["train-corpus"]
    return extra


expc, SnakeMake = mk_expcomb(EXPERIMENTS, eval, extra_pk)


@expc.mk_train
@click.argument("train_corpus", type=click.Path())
@click.argument("model", type=click.Path())
def train(train_corpus, model):
    return ExpPathInfo(corpus=train_corpus, model_full=model)


@expc.mk_test
@click.argument("test_corpus", type=click.Path())
@click.argument("guess", type=click.Path())
@click.option("--model", type=click.Path())
def test(test_corpus, guess, model):
    return ExpPathInfo(corpus=test_corpus, guess_full=guess, model_full=model)


@expc.exp_apply_cmd
@click.argument("db", type=TinyDBParam())
@click.argument("guess", type=click.Path())
@click.argument("gold", type=click.Path())
@click.argument("kvs", nargs=-1)
def eval(exp, db, guess, gold, kvs):
    from expcomb.score import proc_score

    measures = eval_func(pjoin(gold, "corpus.key"), guess)
    proc_score(exp, db, measures, gold, **dict((kv.split("=") for kv in kvs)))
