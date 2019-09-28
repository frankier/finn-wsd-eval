from expcomb.table.spec import (
    DimGroups,
    SqTableSpec,
    SumTableSpec,
    CatValGroup,
    LookupGroupDisplay,
    UnlabelledMeasure,
    box_highlight,
)
from expcomb.filter import SimpleFilter, OrFilter


def tf_group(name):
    return CatValGroup(name, [False, True])


LESK_VECS = LookupGroupDisplay(
    CatValGroup("opts,vec", ["fasttext", "numberbatch", "double"]),
    {"fasttext": "fastText", "numberbatch": "Numberbatch", "double": "Concat"},
)
NN_VECS = LookupGroupDisplay(
    CatValGroup(
        "opts,vec", ["fasttext", "numberbatch", "word2vec", "double", "triple"]
    ),
    {
        "fasttext": "fastText",
        "numberbatch": "Numberbatch",
        "word2vec": "Word2Vec",
        "double": "Concat 2",
        "triple": "Concat 3",
    },
)
NN_MEANS = LookupGroupDisplay(
    CatValGroup(
        "opts,mean", ["pre_sif_mean", "normalized_mean", "catp3_mean", "catp4_mean"]
    ),
    {
        "normalized_mean": "AWE",
        "catp3_mean": "CATP3",
        "catp4_mean": "CATP4",
        "pre_sif_mean": "pre-SIF",
    },
)

LESK_MEANS = LookupGroupDisplay(
    CatValGroup(
        "opts,mean", ["pre_sif_mean", "unnormalized_mean", "catp3_mean", "catp4_mean"]
    ),
    {
        "unnormalized_mean": "AWE",
        "catp3_mean": "CATP3",
        "catp4_mean": "CATP4",
        "pre_sif_mean": "pre-SIF",
    },
)

TRAIN_CORPORA = LookupGroupDisplay(
    CatValGroup("train-corpus", ["eurosense-train", "stiff-train"]),
    {"eurosense-train": "Eurosense trained", "stiff-train": "STIFF trained"},
)

TEST_CORPORA = LookupGroupDisplay(
    CatValGroup("test-corpus", ["eurosense-test", "stiff-test"]),
    {"eurosense-test": "Eurosense tested", "stiff-test": "STIFF tested"},
)

DEV_CORPORA = LookupGroupDisplay(
    CatValGroup("test-corpus", ["eurosense-dev", "stiff-dev"]),
    {"eurosense-dev": "Eurosense tested", "stiff-dev": "STIFF tested"},
)

LESK_SQUARE_SPEC = SqTableSpec(
    DimGroups(
        [
            # LookupGroupDisplay(
            # tf_group("opts,use_freq"), {False: "No freq", True: "Freq"}
            # ),
            DEV_CORPORA,
            LESK_VECS,
            LESK_MEANS,
        ],
        0,
    ),
    DimGroups(
        [
            LookupGroupDisplay(
                tf_group("opts,expand"), {False: "No expand", True: "Expand"}
            ),
            LookupGroupDisplay(
                tf_group("opts,wn_filter"), {False: "No filter", True: "Filter"}
            ),
        ]
    ),
    UnlabelledMeasure("F1"),
    box_highlight,
)

TABLES = [
    (
        "full_sum_table",
        SumTableSpec(DimGroups([TEST_CORPORA, TRAIN_CORPORA]), UnlabelledMeasure("F1")),
        OrFilter(
            SimpleFilter("Baseline"),
            SimpleFilter("Knowledge", "UKB"),
            SimpleFilter("Supervised", "SupWSD"),
            SimpleFilter("Supervised", "AWE-NN", mean="normalized_mean", vec="double"),
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
            SimpleFilter(
                "Knowledge",
                "Cross-lingual Lesk",
                **{
                    "use_freq": True,
                    "vec": "double",
                    "expand": False,
                    "wn_filter": False,
                    "mean": "unnormalized_mean",
                }
            ),
        ),
    ),
    (
        "lesk_square",
        LESK_SQUARE_SPEC,
        SimpleFilter("Knowledge", "Cross-lingual Lesk", **{"use_freq": False}),
    ),
    (
        "lesk_freq_square",
        LESK_SQUARE_SPEC,
        SimpleFilter("Knowledge", "Cross-lingual Lesk", **{"use_freq": True}),
    ),
    (
        "nn_awe_square",
        SqTableSpec(
            DimGroups([TRAIN_CORPORA, DEV_CORPORA, NN_MEANS], 1),
            DimGroups([NN_VECS]),
            UnlabelledMeasure("F1"),
            box_highlight,
        ),
        SimpleFilter("Supervised", "AWE-NN"),
    ),
]
