import click
from plumbum import local
from plumbum.cmd import git, pipenv
from shutil import copyfile
import os


@click.command()
@click.option("--gpu/--no-gpu")
def ctx2vec(gpu):
    os.makedirs("systems", exist_ok=True)
    with local.cwd("systems"):
        git("clone", "https://github.com/orenmel/context2vec.git")

    subdir = "gpu" if gpu else "nogpu"
    for fn in ["Pipfile", "Pipfile.lock"]:
        copyfile(
            "support/context2vec/{}/{}".format(subdir, fn),
            "systems/context2vec/{}".format(fn),
        )

    with local.cwd("systems/context2vec"):
        pipenv("install")


if __name__ == "__main__":
    ctx2vec()
