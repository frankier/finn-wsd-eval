import click

from finntk.wordnet.reader import fiwn_encnt
from finntk.emb.concat import ft_nb_multispace
from finntk.emb.fasttext import multispace as fasttext_multispace
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.wsd.lesk_emb import MultilingualLesk
from wsdeval.means import get_mean
from wsdeval.utils import write_lemma


@click.command()
@click.argument("vec")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--wn-filter/--no-wn-filter")
@click.option("--expand/--no-expand")
@click.option("--use-freq/--no-use-freq")
def lesk(vec, mean, inf, keyout, wn_filter, expand, use_freq):
    if vec == "fasttext":
        multispace = fasttext_multispace
    elif vec == "numberbatch":
        multispace = numberbatch_multispace
    elif vec == "double":
        multispace = ft_nb_multispace
    else:
        assert False
    return wordvec_lesk(
        get_mean(mean, vec), multispace, wn_filter, expand, use_freq, inf, keyout
    )


def wordvec_lesk(aggf, multispace, wn_filter, expand, use_freq, inf, keyout):
    from wsdeval.sup_corpus import iter_instances, norm_wf_lemma_of_tokens

    lesk = MultilingualLesk(multispace, aggf, wn_filter, expand, use_freq)
    for inst_id, lemma_pos, (be, he, af) in iter_instances(inf):
        lemma, pos = lemma_pos
        ctx = norm_wf_lemma_of_tokens(be + af)
        if use_freq:
            lemmas = fiwn_encnt.lemma_freqs(lemma, pos=pos)
        else:
            lemmas = fiwn_encnt.lemmas(lemma, pos=pos)
        lemma, dist = lesk.disambg_one(lemmas, ctx)
        write_lemma(keyout, inst_id, lemma)


if __name__ == "__main__":
    lesk()
