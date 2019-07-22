import click
from plumbum import local
from plumbum.cmd import git, pipenv
from shutil import copyfile
import os
from finntk.utils import urlretrieve
from zipfile import ZipFile


@click.command()
@click.option("--gpu/--no-gpu")
@click.option("--skip-pip/--no-skip-pip")
def ctx2vec(gpu, skip_pip):
    os.makedirs("systems", exist_ok=True)
    with local.cwd("systems"):
        git("clone", "https://github.com/orenmel/context2vec.git")

    if not skip_pip:
        subdir = "gpu" if gpu else "nogpu"
        for fn in ["Pipfile", "Pipfile.lock"]:
            copyfile(
                "support/context2vec/{}/{}".format(subdir, fn),
                "systems/context2vec/{}".format(fn),
            )

    for fn in ["test.py", "train.py"]:
        copyfile(
            "support/context2vec/{}".format(fn),
            "systems/context2vec/context2vec/eval/wsd/{}".format(fn),
        )

    with local.cwd("systems/context2vec"), local.env(PIPENV_IGNORE_VIRTUALENVS="1"):
        if not skip_pip:
            pipenv("install")

        tmp_zipped_model_fn = urlretrieve(
            "https://archive.org/download/ctx2vec-b100-3epoch/ctx2vec-b100-3epoch.zip"
        )
        try:
            tmp_zip = ZipFile(tmp_zipped_model_fn)
            tmp_zip.extractall(".")
        finally:
            os.remove(tmp_zipped_model_fn)


if __name__ == "__main__":
    ctx2vec()
