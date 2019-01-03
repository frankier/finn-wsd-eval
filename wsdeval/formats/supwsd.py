import re


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


def proc_supwsd(goldkey, supwsd_result_fp, guess_fp):
    for gold_line, supwsd_result in zip(goldkey, iter_supwsd_result(supwsd_result_fp)):
        key = gold_line.split()[0]
        synsets = supwsd_result[1]
        if len(synsets):
            synset = synsets[0][0].decode("utf-8")
        else:
            synset = "U"
        guess_fp.write("{} {}\n".format(key, synset))
