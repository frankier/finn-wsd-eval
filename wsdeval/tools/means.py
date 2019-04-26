from finntk.emb.utils import (
    CATP_3,
    CATP_4,
    catp_mean,
    unnormalized_mean,
    normalized_mean,
    mk_sif_mean,
    pre_sif_mean,
)

from functools import partial
import pickle
from os.path import join as pjoin
import os

EXPANDING_MEANS = {
    "catp3_mean": partial(catp_mean, ps=CATP_3),
    "catp4_mean": partial(catp_mean, ps=CATP_4),
}

NON_EXPANDING_MEANS = {
    "unnormalized_mean": unnormalized_mean,
    "normalized_mean": normalized_mean,
    "pre_sif_mean": pre_sif_mean,
}

ALL_MEANS = {**EXPANDING_MEANS, **NON_EXPANDING_MEANS}

MEAN_DISPS = {
    "catp3_mean": "CATP3-WE",
    "catp4_mean": "CATP4-WE",
    "sif_mean": "SIF-WE",
    "pre_sif_mean": "preSIF-WE",
    "unnormalized_mean": "AWE",
    "normalized_mean": "AWE-norm",
}

pcs_cache = {}


def get_mean(mean, emb_name):
    if mean == "sif_mean":
        if emb_name not in pcs_cache:
            vec = pickle.load(
                open(
                    pjoin(
                        os.environ.get("WSDEVAL_SUPPORT", "support"), "sif/{}.pkl"
                    ).format(emb_name),
                    "rb",
                )
            )
            pcs_cache[emb_name] = vec[0]
        return mk_sif_mean(pcs_cache[emb_name])
    else:
        return ALL_MEANS[mean]
