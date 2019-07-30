import click
from finntk.wordnet.reader import fiwn
from finntk.wordnet.utils import fi2en_post
from os.path import join as pjoin, exists
from os import makedirs
import pickle

from wsdeval.formats.sup_corpus import iter_instances_grouped, next_key


@click.group()
def limits():
    pass


@limits.command("train-ceil")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("model", type=click.Path())
def train_ceil(inf, keyin, model):
    makedirs(model)
    for item_pos, cnt, it in iter_instances_grouped(inf):
        item_pos_str = ".".join(item_pos)
        all_synset_ids = set()
        for inst_id, _ in it:
            key_id, synset_ids = next_key(keyin)
            assert inst_id == key_id
            all_synset_ids.update(synset_ids)
        with open(pjoin(model, item_pos_str), "wb") as outf:
            pickle.dump(all_synset_ids, outf, protocol=4)


@limits.command("test-ceil")
@click.argument("model", type=click.Path())
@click.argument("inf", type=click.File("rb"))
@click.argument("keyin", type=click.File("r"))
@click.argument("keyout", type=click.File("w"))
def test_ceil(model, inf, keyin, keyout):
    for item_pos, cnt, it in iter_instances_grouped(inf):
        item_pos_str = ".".join(item_pos)
        model_path = pjoin(model, item_pos_str)
        if exists(model_path):
            with open(model_path, "rb") as inf:
                all_synset_ids = pickle.load(inf)
        else:
            all_synset_ids = []
        for inst_id, _ in it:
            key_id, gold_synset_ids = next_key(keyin)
            prediction = set(gold_synset_ids) & set(all_synset_ids)
            if not prediction:
                prediction = ["U"]
            keyout.write("{} {}\n".format(inst_id, " ".join(prediction)))


@limits.command("floor")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
def floor(inf, keyout):
    for (word, pos), cnt, it in iter_instances_grouped(inf):
        lemmas = fiwn.lemmas(word, pos=pos)
        if len(lemmas) == 1:
            for inst_id, _ in it:
                keyout.write(
                    "{} {}\n".format(
                        inst_id, fi2en_post(fiwn.ss2of(lemmas[0].synset()))
                    )
                )
        else:
            for inst_id, _ in it:
                keyout.write("{} U\n".format(inst_id))


if __name__ == "__main__":
    limits()
