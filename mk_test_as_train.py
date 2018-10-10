from os import makedirs, symlink, remove
from os.path import abspath, lexists
import click
from stiff.eval import get_eval_paths


@click.command()
@click.argument("corpus_in", type=click.Path())
@click.argument("corpus_out", type=click.Path())
def main(corpus_in, corpus_out):
    root_in, paths_in = get_eval_paths(corpus_in)
    makedirs(corpus_out, exist_ok=True)
    root_out, paths_out = get_eval_paths(corpus_out)
    for partition in ["test", "train"]:
        for tag in paths_in[partition]:
            dest = paths_out[partition][tag]
            if lexists(dest):
                remove(dest)
            src = abspath(paths_in["test"][tag])
            symlink(src, dest)


if __name__ == '__main__':
    main()
