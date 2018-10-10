import click

from finntk.wordnet.reader import fiwn
from finntk.emb.concat import ft_nb_multispace
from finntk.emb.fasttext import multispace as fasttext_multispace
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.wsd.lesk_emb import MultilingualLesk
from stiff.utils.xml import iter_sentences
from means import ALL_MEANS
from utils import lemmas_from_instance, write_lemma


@click.command()
@click.argument("vec")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--wn-filter/--no-wn-filter")
@click.option("--expand/--no-expand")
def lesk(vec, mean, inf, keyout, wn_filter, expand):
    if vec == "fasttext":
        multispace = fasttext_multispace
    elif vec == "numberbatch":
        multispace = numberbatch_multispace
    else:
        multispace = ft_nb_multispace
    return wordvec_lesk(
        ALL_MEANS[mean],
        multispace,
        wn_filter,
        expand,
        inf,
        keyout
    )


def wordvec_lesk(aggf, multispace, wn_filter, expand, inf, keyout):
    lesk = MultilingualLesk(
        multispace,
        aggf,
        wn_filter,
        expand
    )
    for sent in iter_sentences(inf):
        for instance in sent.xpath("instance"):
            context = []
            word_tags = sent.xpath("/wf|instance")
            for tag in word_tags:
                if tag == instance:
                    continue
                context.append((
                    instance.text.lower(),
                    instance.attrib["lemma"].lower()
                ))
            inst_id = instance.attrib["id"]
            _lemma_str, _pos, lemmas = lemmas_from_instance(fiwn, instance)
            lemma, dist = lesk.disambg_one(
                lemmas,
                context,
            )
            write_lemma(keyout, inst_id, lemma)


if __name__ == '__main__':
    lesk()
