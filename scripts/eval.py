import click
from os import makedirs
from tinydb import TinyDB
from wsdeval.exps.base import ExpPathInfo
from wsdeval.exps.confs import EXPERIMENTS


def parse_opts(opts):
    opt_dict = {}
    for opt in opts:
        k, v = opt.split("=")
        if v in ["True", "False"]:
            py_v = v == "True"
        elif v == "None":
            py_v = None
        else:
            try:
                py_v = int(v)
            except ValueError:
                py_v = v
        opt_dict[k] = py_v
    return opt_dict


class TinyDBParam(click.Path):
    def convert(self, value, param, ctx):
        if isinstance(value, TinyDB):
            return value
        path = super().convert(value, param, ctx)
        return TinyDB(path).table("results")


@click.group(chain=True)
@click.pass_context
@click.option("--filter")
def eval(ctx, filter):
    filter_l1, filter_l2, opts = None, None, {}
    filter_bits = filter.split(" ")
    if len(filter_bits) >= 1:
        filter_l1 = filter_bits[0]
    if len(filter_bits) >= 2:
        filter_l2 = filter_bits[1]
    if len(filter_bits) >= 3:
        opts = parse_opts(filter_bits[2:])
    ctx.ensure_object(dict)
    ctx.obj["filter"] = (filter_l1, filter_l2, opts)


@eval.command("train")
@click.pass_context
@click.argument("train_corpus", type=click.Path())
@click.argument("models_dir", type=click.Path())
def train(ctx, train_corpus, models_dir):
    makedirs(models_dir, exist_ok=True)
    for exp_group in EXPERIMENTS:
        path_info = ExpPathInfo(train_corpus, "", models_dir)
        exp_group.train_all(path_info, *ctx.obj["filter"])


@eval.command("test")
@click.pass_context
@click.argument("test_corpus", type=click.Path())
@click.argument("db", type=TinyDBParam())
@click.argument("guess_dir", type=click.Path())
@click.argument("models_dir", type=click.Path(), required=False)
def test(ctx, test_corpus, db, guess_dir, models_dir):
    makedirs(guess_dir, exist_ok=True)
    for exp_group in EXPERIMENTS:
        path_info = ExpPathInfo(test_corpus, guess_dir, models_dir)
        exp_group.run_all(db, path_info, *ctx.obj["filter"])


if __name__ == "__main__":
    eval()
