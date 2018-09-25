import os
import click
from plumbum import local
from plumbum.cmd import git, java, mvn
from string import Template
from os.path import join as pjoin


SUPWSD_JAR = "target/supwsd-toolkit-1.0.0.jar"


@click.group()
def supwsd():
    pass


@supwsd.command()
def prepare():
    fetch()
    conf()


@supwsd.command()
def fetch():
    os.makedirs("systems", exist_ok=True)
    with local.cwd("systems"):
        git("clone", "https://github.com/frankier/supWSD.git")
        with local.cwd("supWSD"):
            git("checkout", "fixes-sep-24-1")
            mvn("package")


@supwsd.command()
def conf():
    from finntk.wordnet.reader import fiwn_resman

    fiwn_path = fiwn_resman.get_res("")
    for src_fn, dst_fn in [
            ("jwnl-properties.xml", "resources/wndictionary/prop.xml"),
            ("supconfig.xml", "supconfig.xml")
    ]:
        content = (
            Template(open("support/supWSD/{}.tmpl".format(src_fn)).read()).substitute({
                "FIWN_PATH": fiwn_path,
            })
        )

        with open("systems/supWSD/{}".format(dst_fn), "w") as dst_f:
            dst_f.write(content)


@supwsd.command()
@click.argument("inf")
@click.argument("keyin")
def train(inf, keyin):
    with local.cwd("systems/supWSD"):
        inf_path = pjoin("../../", inf)
        keyin_path = pjoin("../../", keyin)
        print(
            "java", "-jar", SUPWSD_JAR, "train",
            "supconfig.xml", inf_path, keyin_path)
        java(
            "-jar", SUPWSD_JAR, "train",
            "supconfig.xml", inf_path, keyin_path
        )


@supwsd.command()
@click.argument("inf")
@click.argument("resultout")
def test(inf, resultout):
    with local.cwd("systems/supWSD"):
        inf_path = pjoin("../../", inf)
        resultout_path = pjoin("../../", resultout)
        java(
            "-jar", SUPWSD_JAR, "test",
            "supconfig.xml", inf_path, resultout_path
        )


if __name__ == "__main__":
    supwsd()
