from finntk.finnpos import sent_finnpos
from finntk.wordnet.reader import fiwn_encnt
from finntk.emb.numberbatch import multispace as numberbatch_multispace
from finntk.emb.autoextend import mk_lemma_vec
from finntk.wsd.lesk_pp import LeskPP
from finntk.emb.utils import cosine_sim
from stiff.utils.xml import iter_sentences
import click
from wsdeval.tools.means import ALL_MEANS
from wsdeval.formats.wordnet import lemmas_from_instance, write_lemma
import wsdeval.tools.log  # noqa
import logging


logger = logging.getLogger(__name__)


@click.command("lesk-pp")
@click.argument("mean")
@click.argument("inf", type=click.File("rb"))
@click.argument("keyout", type=click.File("w"))
@click.option("--include-wfs/--no-include-wfs")
@click.option("--expand/--no-expand")
@click.option("--exclude-cand/--no-exclude-cand")
@click.option("--score-by")
def lesk_pp(mean, inf, keyout, include_wfs, expand, exclude_cand, score_by):
    aggf = ALL_MEANS[mean]
    lesk_pp = LeskPP(numberbatch_multispace, aggf, False, expand)
    for sent_idx, sent in enumerate(iter_sentences(inf)):
        if include_wfs:
            instances = sent.xpath("instance|wf")
            sent = [inst.text for inst in instances]
            tagged_sent = sent_finnpos(sent)
        else:
            instances = sent.xpath("instance")
            tagged_sent = None
        sent_lemmas = []
        instance_ids = []
        # XXX: SHOULD add wfs too! (equiv to wn_filter)
        for idx, instance in enumerate(instances):
            if instance.tag == "wf":
                lemma_str = tagged_sent[idx][1]
                lemmas = []
            else:
                lemma_str, _pos, lemmas = lemmas_from_instance(fiwn_encnt, instance)
            sent_lemmas.append((lemma_str, lemmas))
            if instance.tag == "instance":
                instance_ids.append(instance.attrib["id"])
        disambg_order = sorted(
            (len(lemmas), idx)
            for idx, (lemma_str, lemmas) in enumerate(sent_lemmas)
            if len(lemmas) > 0
        )
        for ambiguity, lemma_idx in disambg_order:
            if ambiguity <= 1:
                continue
            lemma_str, lemmas = sent_lemmas[lemma_idx]

            # XXX: Should context_vec exclude the word being disambiguated
            context_vec = lesk_pp.mk_ctx_vec(
                sent_lemmas, *([lemma_idx] if exclude_cand else [])
            )
            if context_vec is None:
                logger.debug("No context vec, backing off to MFS")
                # Back off to MFS
                sent_lemmas[lemma_idx] = (lemma_str, [lemmas[0]])
            else:
                logger.debug(f"Got context vec {context_vec}")
                best_lemma = None
                best_score = -2
                for lemma in lemmas:
                    logger.debug(f"Considering lemma: {lemma}")
                    defn_vec = lesk_pp.mk_defn_vec(lemma)
                    logger.debug(f"Got defn_vec: {defn_vec}")
                    if defn_vec is None:
                        defn_ctx_score = 0
                    else:
                        defn_ctx_score = cosine_sim(defn_vec, context_vec)
                    try:
                        lemma_vec = mk_lemma_vec(lemma)
                    except KeyError:
                        # XXX: Is this reasonable, or should there be a penalty?
                        lemma_ctx_score = defn_ctx_score
                    else:
                        logger.debug(f"Got lemma_vec: {lemma_vec}")
                        lemma_ctx_score = cosine_sim(lemma_vec, context_vec)
                    if score_by == "both":
                        score = defn_ctx_score + lemma_ctx_score
                    elif score_by == "defn":
                        score = defn_ctx_score
                    elif score_by == "lemma":
                        score = lemma_ctx_score
                    else:
                        assert False
                    logger.debug(
                        f"Score: {score} ({defn_ctx_score} + {lemma_ctx_score})"
                    )
                    if score > best_score:
                        best_lemma = lemma
                        best_score = score
                sent_lemmas[lemma_idx] = (lemma_str, [best_lemma])
        instance_sent_lemmas = (x for x in sent_lemmas if len(x[1]) > 0)
        for (lemma_str, lemmas), inst_id in zip(instance_sent_lemmas, instance_ids):
            if lemmas[0] is None:
                continue
            write_lemma(keyout, inst_id, lemmas[0])


if __name__ == "__main__":
    lesk_pp()
