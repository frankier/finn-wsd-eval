import os
import click
from expcomb.cmd import mk_expcomb
from expcomb.utils import TinyDBParam
from expcomb.sigtest.bootstrap import (
    simple_compare_resampled,
    simple_create_schedule,
    simple_resample,
    Bootstrapper,
)
from wsdeval.exps.base import ExpPathInfo
from wsdeval.exps.confs import EXPERIMENTS
from wsdeval.exps.utils import score as eval_func
from wsdeval.tables import TABLES
from os.path import join as pjoin
import logging

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def extra_pk(doc):
    extra = {"test-corpus": doc["test-corpus"]}
    if "train-corpus" in doc:
        extra["train-corpus"] = doc["train-corpus"]
    return extra


expc, SnakeMake = mk_expcomb(EXPERIMENTS, eval, extra_pk, tables=TABLES)


@expc.mk_train
@click.argument("train_corpus", type=click.Path())
@click.argument("model", type=click.Path())
@click.option("--multi/--single")
def train(train_corpus, model, multi):
    kwargs = {}
    if multi:
        kwargs["models"] = model
    else:
        kwargs["model_full"] = model
    return ExpPathInfo(corpus=train_corpus, **kwargs)


@expc.mk_test
@click.argument("test_corpus", type=click.Path())
@click.argument("guess", type=click.Path())
@click.option("--model", type=click.Path())
@click.option("--multi/--single")
def test(test_corpus, guess, model, multi):
    kwargs = {}
    if multi:
        kwargs["models"] = model
        kwargs["guess"] = guess
    else:
        kwargs["model_full"] = model
        kwargs["guess_full"] = guess
    return ExpPathInfo(corpus=test_corpus, **kwargs)


@expc.exp_apply_cmd
@click.argument("db", type=TinyDBParam())
@click.argument("guess", type=click.Path())
@click.argument("gold", type=click.Path())
@click.argument("kvs", nargs=-1)
def eval(exp, db, guess, gold, kvs):
    from expcomb.score import proc_score

    measures = eval_func(pjoin(gold, "corpus.sup.key"), guess)
    proc_score(exp, db, measures, guess, gold, **dict((kv.split("=") for kv in kvs)))


def hmean(x, y):
    if x == 0 or y == 0:
        return 0
    return 2 * x * y / (x + y)


def resampled_f1_score(resample, line_map, guess_map, gold_map):
    tp = 0
    fp = 0
    for sample_idx in resample:
        sample_key = line_map[sample_idx]
        guesses = guess_map.get(sample_key, [])
        if guesses:
            local_tp = 0
            local_fp = 0
            len_guesses = len(guesses)
            gold_set = gold_map[sample_key]
            for guess in guesses:
                if guess in gold_set:
                    local_tp += 1
                else:
                    local_fp += 1
            tp += local_tp / len_guesses
            fp += local_fp / len_guesses
    all_p = tp + fp
    p = tp / all_p if all_p > 0 else 0
    r = tp / len(resample)
    return hmean(p, r)


def results_to_map(results_lines):
    results_map = {}
    for line in results_lines:
        inst_key, sense_keys = line.strip().split(" ", 1)
        # XXX: Conceptually it may seem as if we should treat "U" as [], but
        # that's not the way Scorer.java works. This means that systems that
        # guess U will get a lower F1 score than systems that do not output
        # anything.
        results_map[inst_key] = sense_keys.split(" ")
    return results_map


def create_sample_maps(gold, guess):
    line_map = []
    guess_lines = open(guess).readlines()
    guess_map = results_to_map(guess_lines)
    gold_lines = open(gold).readlines()
    gold_map = results_to_map(gold_lines)
    for line in gold_lines:
        inst_key, _sense_key = line.strip().split(" ", 1)
        line_map.append(inst_key)
    return line_map, guess_map, gold_map


class WSDEvalBootstrapper(Bootstrapper):
    def score_one(self, gold, guess):
        score_dict = eval_func(gold, guess)
        scorer_score = float(score_dict["F1"].rstrip("%"))

        line_map, guess_map, gold_map = create_sample_maps(gold, guess)
        our_score = (
            resampled_f1_score(
                range(len(open(gold).readlines())), line_map, guess_map, gold_map
            )
            * 100
        )
        assert (
            abs(scorer_score - our_score) < 0.1
        ), f"Scores for {gold} and {guess}: {scorer_score} and {our_score} should have less than 0.1 difference"

        return our_score

    def create_score_dist(self, gold, guess, schedule):
        line_map, guess_map, gold_map = create_sample_maps(gold, guess)
        dist = []
        for resample in schedule:
            dist.append(
                resampled_f1_score(resample, line_map, guess_map, gold_map) * 100
            )
        return dist


bootstrapper = WSDEvalBootstrapper()

simple_compare_resampled()
simple_create_schedule(bootstrapper)
simple_resample(bootstrapper, extra_pk)
