import os
from os.path import join as pjoin
import lzma
import click
from typing import IO


@click.command()
@click.argument("indir", type=click.Path())
@click.argument("outf", type=click.File("wb"))
def conll17_to_plain(indir: str, outf: IO):
    for filename in sorted(os.listdir(indir)):
        if not filename.endswith(".xz"):
            continue
        full_path = pjoin(indir, filename)
        inf = lzma.open(full_path)
        for line in inf:
            if line == b"\n":
                outf.write(b"\n")
            elif line[0] != b"#"[0]:
                outf.write(line.split(b"\t", 2)[1])
                outf.write(b" ")


if __name__ == "__main__":
    conll17_to_plain()
