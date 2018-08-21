import os
import click
from plumbum import local
from plumbum.cmd import git, wget, tar, rm, bash, python
from shutil import copyfile


@click.group()
def ims():
    pass


@ims.command()
def fetch():
    os.makedirs("systems", exist_ok=True)
    with local.cd("systems"):
        git("clone", "https://github.com/frankier/ims.git")
        with local.cd("ims"):
            wget("http://www.comp.nus.edu.sg/~nlp/sw/lib.tar.gz")
            tar("-xvzf", "lib.tar.gz")
            rm("lib.tar.gz")

    copyfile("support/ims/test_ims.bash", "systems/ims/")
    copyfile("support/ims/train_ims.bash", "systems/ims/")


@ims.command()
@click.argument("inf")
@click.argument("keyin")
@click.argument("modelout")
def train(inf, keyin, modelout):
    bash("systems/ims/train_one.bash", inf, keyin, modelout)


@ims.command()
@click.argument("modelin")
@click.argument("inf")
@click.argument("resultout")
def test(modelin, inf, resultout):
    bash("systems/ims/test_ims.bash", modelin, inf, "imsresult")
    python(__file__, "fixup-keyout", "imsresult", resultout)


@ims.command("fixup-keyout")
@click.argument("keyin", type=click.File("r"))
@click.argument("keyout", type=click.File("w"))
def fixup_keyout(keyin, keyout):
    lines = sorted((l[1:] for l in open(keyin)))
    for line in lines:
        keyout.write(line)
