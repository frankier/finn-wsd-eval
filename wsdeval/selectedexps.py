from expcomb.filter import AndFilter, SimpleFilter


def corpus(corpusseg):
    return corpusseg.split("-", 1)[0]


def seg(corpusseg):
    return corpusseg.split("-", 1)[1]


class TrainSegFilter:
    def doc_included(self, d_path, d_opts):
        return seg(d_opts["train-corpus"]) == "train"


class InCorpusFilter:
    def doc_included(self, d_path, d_opts):
        res = corpus(d_opts["train-corpus"]) == corpus(d_opts["test-corpus"])
        return res


class OutCorpusFilter:
    def doc_included(self, d_path, d_opts):
        res = corpus(d_opts["train-corpus"]) != corpus(d_opts["test-corpus"])
        return res


UNSUP_SELECTED = [
    ("1st", SimpleFilter("Baseline", "1st")),
    ("Rand", SimpleFilter("Baseline", "Rand")),
    (
        "UKB",
        SimpleFilter("Knowledge", "UKB", **{"extract_extra": False, "use_freq": False}),
    ),
    (
        "UKB freq",
        SimpleFilter("Knowledge", "UKB", **{"extract_extra": False, "use_freq": True}),
    ),
    (
        "Lesk",
        SimpleFilter(
            "Knowledge",
            "Cross-lingual Lesk",
            **{
                "use_freq": False,
                "vec": "double",
                "expand": True,
                "wn_filter": False,
                "mean": "catp3_mean",
            }
        ),
    ),
    (
        "Lesk freq",
        SimpleFilter(
            "Knowledge",
            "Cross-lingual Lesk",
            **{
                "use_freq": True,
                "vec": "double",
                "expand": False,
                "wn_filter": False,
                "mean": "normalized_mean",
            }
        ),
    ),
    (
        "Lesk++",
        SimpleFilter(
            "Knowledge",
            "Lesk++",
            **{
                "expand": False,
                "mean": "normalized_mean",
                "exclude_cand": True,
                "score_by": "both",
            }
        ),
    ),
]

SUP_SELECTED = [
    ("SupWSD", SimpleFilter("Supervised", "SupWSD", vec="fasttext", sur_words=True)),
    (
        "AWE-NN",
        SimpleFilter("Supervised", "AWE-NN", mean="normalized_mean", vec="double"),
    ),
]

INOUT = [("in", InCorpusFilter()), ("out", OutCorpusFilter())]

SUP_INOUT_SELECTED = [
    (label + " " + inout, AndFilter(filter, TrainSegFilter(), inout_filter))
    for label, filter in SUP_SELECTED
    for inout, inout_filter in INOUT
]

LIMITS_SELECTED = [
    ("Unambg", SimpleFilter("Limits", "Floor")),
    ("1st", SimpleFilter("Baseline", "1st")),
    ("Rand", SimpleFilter("Baseline", "Rand")),
]

for inout, inout_filter in INOUT[::-1]:
    LIMITS_SELECTED += [
        (
            "InstKnown " + inout,
            AndFilter(SimpleFilter("Limits", "CeilInst"), inout_filter),
        ),
        (
            "TokKnown " + inout,
            AndFilter(SimpleFilter("Limits", "CeilTok"), inout_filter),
        ),
    ]
