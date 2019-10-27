from expcomb.table.spec import (
    SelectDimGroups,
    DimGroups,
    SqTableSpec,
    SumTableSpec,
    CatValGroup,
    LookupGroupDisplay,
    UnlabelledMeasure,
    box_highlight,
)
from expcomb.filter import SimpleFilter
from .selectedexps import UNSUP_SELECTED, SUP_SELECTED, SUP_INOUT_SELECTED


def tf_group(name):
    return CatValGroup(name, [False, True])


LESK_VECS = LookupGroupDisplay(
    CatValGroup("vec", ["fasttext", "numberbatch", "double"]),
    {"fasttext": "fastText", "numberbatch": "Numberbatch", "double": "Concat"},
)
NN_VECS = LookupGroupDisplay(
    CatValGroup("vec", ["fasttext", "numberbatch", "word2vec", "double", "triple"]),
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
        "mean", ["pre_sif_mean", "normalized_mean", "catp3_mean", "catp4_mean"]
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
        "mean", ["pre_sif_mean", "normalized_mean", "catp3_mean", "catp4_mean"]
    ),
    {
        "normalized_mean": "AWE",
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
            # tf_group("use_freq"), {False: "No freq", True: "Freq"}
            # ),
            DEV_CORPORA,
            LESK_VECS,
            LESK_MEANS,
        ],
        div_idx=0,
    ),
    DimGroups(
        [
            LookupGroupDisplay(
                tf_group("expand"), {False: "No expand", True: "Expand"}
            ),
            LookupGroupDisplay(
                tf_group("wn_filter"), {False: "No filter", True: "Filter"}
            ),
        ]
    ),
    UnlabelledMeasure("F1"),
    box_highlight,
)

UNSUP_GROUPS = SelectDimGroups(*(UNSUP_SELECTED + SUP_INOUT_SELECTED))
SUP_GROUPS = SelectDimGroups(*SUP_SELECTED)

TABLES = [
    (
        "unsup_sum_table",
        SumTableSpec(UNSUP_GROUPS, DimGroups([TEST_CORPORA]), UnlabelledMeasure("F1")),
    ),
    (
        "sup_sum_table",
        SumTableSpec(
            SUP_GROUPS,
            DimGroups([TRAIN_CORPORA, TEST_CORPORA]),
            UnlabelledMeasure("F1"),
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
        "supwsd_square",
        SqTableSpec(
            DimGroups([TRAIN_CORPORA, DEV_CORPORA]),
            SelectDimGroups(
                ("fasttext", SimpleFilter(sur_words=False, vec="fasttext")),
                ("word2vec", SimpleFilter(sur_words=False, vec="word2vec")),
                ("-s", SimpleFilter(sur_words=True, vec=None)),
                ("fasttext-s", SimpleFilter(sur_words=True, vec="fasttext")),
                ("word2vec-s", SimpleFilter(sur_words=True, vec="word2vec")),
            ),
            UnlabelledMeasure("F1"),
            box_highlight,
        ),
        SimpleFilter("Supervised", "SupWSD"),
    ),
    (
        "nn_awe_square",
        SqTableSpec(
            DimGroups([TRAIN_CORPORA, DEV_CORPORA, NN_MEANS], div_idx=1),
            DimGroups([NN_VECS]),
            UnlabelledMeasure("F1"),
            box_highlight,
        ),
        SimpleFilter("Supervised", "AWE-NN"),
    ),
]
