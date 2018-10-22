from plumbum import local
from plumbum.cmd import wget, unzip, rm
import click
import os


@click.command()
def sif():
    sif = "support/sif"
    os.makedirs(sif, exist_ok=True)
    with local.cwd(sif):
        wget("https://github.com/frankier/finn-wsd-eval/releases/download/bins/sif.zip")
        unzip("sif.zip")
        rm("sif.zip")


if __name__ == "__main__":
    sif()
