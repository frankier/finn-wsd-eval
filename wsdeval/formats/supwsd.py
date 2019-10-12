import re
from itertools import zip_longest


CHUNK_SIZE = 4096
TAG_REGEX = re.compile(
    br" ?<(?P<START_TAG>.)(?P<SYNSETS>[^>]*)>(?P<WF>.)</(?P<END_TAG>.)>"
)


def proc_match(match):
    groups = match.groupdict()
    assert groups["START_TAG"] == groups["END_TAG"]
    synsets = []
    synsets_str = groups["SYNSETS"].strip(b" ")
    if synsets_str:
        synsets_bits = synsets_str.split(b" ")
        for synset in synsets_bits:
            synset_id, weight_str = synset.split(b"|")
            weight = float(weight_str)
            synsets.append((synset_id, weight))
    return groups["START_TAG"], synsets, groups["WF"]


def iter_supwsd_result(fp):
    buffer = b""
    done = False
    while not done:
        chunk = fp.read(CHUNK_SIZE)
        if not chunk:
            done = True
        else:
            buffer += chunk
        while len(buffer) > 0:
            match = TAG_REGEX.match(buffer)
            if not match:
                break
            yield proc_match(match)
            end_pos = match.end(0)
            if end_pos != len(buffer):
                assert buffer[end_pos] in b" \n"
            buffer = buffer[end_pos + 1 :]
        if done:
            assert len(buffer) == 0


def zip_equal(*iterables, names=None):
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            msg = "Iterables have different lengths. "
            if names is None:
                msg += "Indexes: {} depleted before rest".format(
                    ", ".join(str(idx) for idx, c in enumerate(combo) if c is sentinel)
                )
            else:
                msg += "Columns: {} depleted before rest".format(
                    ", ".join(
                        names[idx] for idx, c in enumerate(combo) if c is sentinel
                    )
                )
            raise ValueError(msg)
        yield combo


def proc_supwsd(goldkey, supwsd_result_fp, guess_fp):
    from finntk.wordnet.reader import fiwn
    from finntk.wordnet.utils import fi2en_post
    from nltk.corpus.reader.wordnet import WordNetError

    # Possible TODO: try picking the most frequent according to what's known
    # here rather than SupWSD's idea of the most frequent. Is it a similar
    # result. Does SupWSD have proper access to frequency data ATM?

    for gold_line, supwsd_result in zip_equal(
        goldkey, iter_supwsd_result(supwsd_result_fp), names=["gold", "supwsd_result"]
    ):
        key = gold_line.split()[0]
        synsets = supwsd_result[1]
        if len(synsets):
            payload = synsets[0][0].decode("utf-8")
            if payload[0].isdigit():
                synset = payload
            else:
                try:
                    lemma = fiwn.lemma_from_key(payload)
                except WordNetError:
                    synset = "U"
                else:
                    synset = fi2en_post(fiwn.ss2of(lemma.synset()))
        else:
            synset = "U"
        guess_fp.write("{} {}\n".format(key, synset))
