import click
from stiff.eval import get_eval_paths
from sup_corpus import iter_lexelts, next_key
from finntk.wordnet.reader import fiwn


def iter_tags(inf, keyin):
    for lexelt, item_pos in iter_lexelts(inf):
        for instance in lexelt.xpath("instance"):
            key_id, synset_ids = next_key(keyin)
            yield item_pos, synset_ids


@click.command()
@click.argument("corpus", type=click.Path())
def main(corpus):
    root, paths = get_eval_paths(corpus)
    seen = set()
    for item_pos, synset_ids in iter_tags(
            open(paths["train"]["sup"], "rb"),
            open(paths["train"]["supkey"])):
        for synset_id in synset_ids:
            seen.add((item_pos, synset_id))
    predictable = 0
    singletons = 0
    total = 0
    for item_pos, synset_ids in iter_tags(open(paths["test"]["sup"], "rb"),
                                          open(paths["test"]["supkey"])):
        lemma, pos = item_pos
        if any((item_pos, synset_id) in seen for synset_id in synset_ids):
            predictable += 1
        if len(fiwn.lemmas(lemma, pos=pos)) == 1:
            singletons += 1
        total += 1
    print(singletons)
    print(predictable)
    print(total)
    print(singletons / total)
    print(predictable / total)


if __name__ == '__main__':
    main()
