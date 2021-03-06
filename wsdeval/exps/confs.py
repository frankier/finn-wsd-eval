import os
from expcomb.models import Exp
from .base import (
    ExpGroup,
    SupExpGroup,
    SupGpuExpGroup,
    EngSupExpGroup,
    EngSupGpuExpGroup,
)
from .exps import (
    AweNn,
    EngAweNn,
    SupWSD,
    Elmo,
    ElmoAllExpGroup,
    BertAllExpGroup,
    Bert2,
    EngBert2,
    baseline,
    lesk_pp,
    ukb,
    Ctx2Vec,
    Ctx2Vec2,
    lesk,
    Floor,
    Ceil,
    Post1stSenseCombExp,
)
from .utils import mk_nick
from wsdeval.tools.means import ALL_MEANS, MEAN_DISPS, NON_EXPANDING_MEANS

EXPERIMENTS = [
    ExpGroup(
        [
            Exp(["Baseline", "Rand"], "first", "FiWN 1st sense", baseline("first")),
            Exp(["Baseline", "1st"], "mfe", "FiWN + PWN 1st sense", baseline("mfe")),
        ]
    ),
    ExpGroup([Floor()]),
    SupExpGroup([Ceil(False), Ceil(True)]),
]

ctx2vec = Ctx2Vec()
EXPERIMENTS.append(SupExpGroup([ctx2vec]))

supwsd_exps = []
for vec, sur_words in [
    (None, True),
    ("word2vec", False),
    ("word2vec", True),
    ("fasttext", False),
    ("fasttext", True),
]:
    supwsd_exps.append(SupWSD(vec, sur_words))
EXPERIMENTS.append(SupExpGroup(supwsd_exps))


MEANS = list(ALL_MEANS.keys())
MEANS.remove("unnormalized_mean")


xlingual_lesk = []
for use_freq in [False, True]:
    for do_expand in [False, True]:
        for vec in ["fasttext", "numberbatch", "double"]:
            lower_vec = vec.lower()
            for mean in MEANS:
                for wn_filter in [False, True]:
                    baseline_args = [lower_vec, mean]
                    nick_extra = ""
                    disp_extra = ""
                    if wn_filter:
                        baseline_args += ["--wn-filter"]
                        nick_extra += ".wn-filter"
                        disp_extra += "+WN-filter"
                    if do_expand:
                        baseline_args += ["--expand"]
                        nick_extra += ".expand"
                        disp_extra += "+expand"
                    if use_freq:
                        baseline_args += ["--use-freq"]
                        nick_extra += ".freq"
                        disp_extra += "+freq"
                    nick = f"lesk.{lower_vec}.{mean}{nick_extra}"
                    mean_disp = "+" + MEAN_DISPS[mean]
                    disp = f"Lesk\\textsubscript{{{vec}{disp_extra}{mean_disp}}}"
                    xlingual_lesk.append(
                        Exp(
                            ["Knowledge", "Cross-lingual Lesk"],
                            nick,
                            disp,
                            lesk(*baseline_args),
                            {
                                "vec": vec,
                                "expand": do_expand,
                                "mean": mean,
                                "wn_filter": wn_filter,
                                "use_freq": use_freq,
                            },
                        )
                    )
EXPERIMENTS.append(ExpGroup(xlingual_lesk))

awe_nn_exps = []
for vec in ["fasttext", "word2vec", "numberbatch", "triple", "double"]:
    for mean in MEANS:
        awe_nn_exps.append(AweNn(vec, mean))
EXPERIMENTS.append(SupExpGroup(awe_nn_exps))

xlingual_awe_nn_exps = []
for vec in ["fasttext", "word2vec", "numberbatch", "triple", "double"]:
    for mean in MEANS:
        xlingual_awe_nn_exps.append(EngAweNn(vec, mean))
EXPERIMENTS.append(EngSupExpGroup(xlingual_awe_nn_exps))


if os.environ.get("USE_SINGLE_LAYER_ELMO"):
    elmo_exps = []
    for layer in (-1, 0, 1, 2):
        elmo_exps.append(Elmo(layer))
    EXPERIMENTS.append(SupGpuExpGroup(elmo_exps))
else:
    EXPERIMENTS.append(ElmoAllExpGroup())

EXPERIMENTS.append(BertAllExpGroup())

EXPERIMENTS.append(SupGpuExpGroup([Bert2()]))
EXPERIMENTS.append(EngSupGpuExpGroup([EngBert2()]))
EXPERIMENTS.append(SupExpGroup([Ctx2Vec2()]))


lesk_pp_exps = []
for score_by in ["both", "defn", "lemma"]:
    for exclude_cand in [False, True]:
        for do_expand in [False, True]:
            for mean in NON_EXPANDING_MEANS:
                lesk_pp_exps.append(
                    Exp(
                        ["Knowledge", "Lesk++"],
                        mk_nick(
                            "lesk_pp",
                            mean,
                            score_by,
                            (exclude_cand, "exclcand"),
                            (do_expand, "expand"),
                        ),
                        "Lesk++ ({} {})".format(MEAN_DISPS[mean], do_expand),
                        lesk_pp(mean, do_expand, exclude_cand, score_by),
                        {
                            "mean": mean,
                            "expand": do_expand,
                            "exclude_cand": exclude_cand,
                            "score_by": score_by,
                        },
                    )
                )
EXPERIMENTS.append(ExpGroup(lesk_pp_exps))


ukb_exps = []
for use_freq in [False, True]:
    for extract_extra in [False, True]:
        ukb_args = ("--ppr_w2w",)
        label_extra = ""
        nick_extra = ""
        if use_freq:
            ukb_args += ("--dict_weight",)
        else:
            ukb_args += ("--dict_noweight",)
            label_extra += "\\textsubscript{nf}"
            nick_extra += ".nf"
        if extract_extra:
            label_extra += "\\textsubscript{+extract}"
            nick_extra += ".extract"
        ukb_exps.append(
            Exp(
                ["Knowledge", "UKB"],
                "ukb" + nick_extra,
                "UKB{}".format(label_extra),
                ukb(True, extract_extra, *ukb_args),
                {"extract_extra": extract_extra, "use_freq": use_freq},
            )
        )
EXPERIMENTS.append(ExpGroup(ukb_exps))

supwsd_1st_exps = []
for supwsd_exp in supwsd_exps:
    supwsd_1st_exps.append(Post1stSenseCombExp(supwsd_exp, True))
EXPERIMENTS.append(SupExpGroup(supwsd_1st_exps))
