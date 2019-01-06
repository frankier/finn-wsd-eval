import click
from datetime import datetime
from os import makedirs
from os.path import join as pjoin
from tinydb import TinyDB
from wsdeval.exps.base import ExpPathInfo
from wsdeval.exps.confs import EXPERIMENTS

DEFAULT_WORK_BASE = "work"


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


@click.command()
@click.argument("db_path", type=click.Path())
@click.argument("filter_l1", required=False)
@click.argument("filter_l2", required=False)
@click.argument("opts", nargs=-1)
@click.option("--train", type=click.Path())
@click.option("--test", type=click.Path())
@click.option("--work-base", type=click.Path(), default=DEFAULT_WORK_BASE)
@click.option("--add-timestamp/--no-add-timestamp", default=True)
def main(
    db_path,
    work_base,
    add_timestamp,
    filter_l1=None,
    filter_l2=None,
    opts=None,
    train=None,
    test=None,
):
    if add_timestamp:
        timestr = datetime.now().isoformat()
        base = pjoin(work_base, timestr)
    else:
        base = work_base
    guess = pjoin(base, "guess")
    models = pjoin(base, "models")
    makedirs(guess, exist_ok=True)
    makedirs(models, exist_ok=True)
    db = TinyDB(db_path).table("results")
    if opts:
        opt_dict = parse_opts(opts)
    else:
        opt_dict = {}
    for exp_group in EXPERIMENTS:
        if train is not None:
            path_info = ExpPathInfo(train, guess, models)
            exp_group.train_all(path_info, filter_l1, filter_l2, opt_dict)
        if test is not None:
            path_info = ExpPathInfo(test, guess, models)
            exp_group.run_all(db, path_info, filter_l1, filter_l2, opt_dict)


if __name__ == "__main__":
    main()
