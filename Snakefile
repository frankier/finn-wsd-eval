## Environment variables
from os.path import join as pjoin
from expcomb.cmd import parse_filter
from wsdeval.expc import SnakeMake

def cnf(name, val):
    globals()[name] = config.setdefault(name, val)

def cnf_list(name, val):
    globals()[name] = config[name].split(",") if name in config else val

# Arguments
cnf("FILTER", "")

# Intermediate dirs
cnf("WORK", "work")
cnf("GUESS", WORK + "/guess")
cnf("RESULTS", WORK + "/results")

#CORPUS_DIRS = [STIFFEVAL, EUROMODELS]
cnf_list("CORPUS_NAMES", ["stiff", "eurosense"])
cnf_list("TRAIN_SEGMENT", ["trainf", "test"])
cnf_list("TEST_SEGMENT", ["dev", "test"])

CORPUS_DIR_MAP = {
    "stiff": lambda: config["STIFFEVAL"],
    "eurosense": lambda: config["EUROSENSEEVAL"],
}

group_at_once_map = SnakeMake.get_group_at_once_map(parse_filter(FILTER))
nick_to_group_nick_map = SnakeMake.get_nick_to_group_nick_map(parse_filter(FILTER))
path_nick_map = SnakeMake.get_path_nick_map(parse_filter(FILTER))

def intersect_nicks(filter, **kwargs):
    if all((k not in filter.opt_dict or filter.opt_dict[k] == v for k, v in kwargs.items())):
        yield from SnakeMake.get_nicks(filter.intersect_opts(**kwargs))


# Utility functions
def all_results():
    filter = parse_filter(FILTER)
    for nick in intersect_nicks(filter, sup=True):
        yield from expand(
            RESULTS + "/" + nick + "/{train_corpus}-{train_seg}/{test_corpus}-{test_seg}.db",
            train_corpus=CORPUS_NAMES,
            train_seg=TRAIN_SEGMENT,
            test_corpus=CORPUS_NAMES,
            test_seg=TEST_SEGMENT
        )
    for nick in intersect_nicks(filter, eng_sup=True):
        yield from expand(
            RESULTS + "/" + nick + "/semcor/{test_corpus}-{test_seg}.db",
            test_corpus=CORPUS_NAMES,
            test_seg=TEST_SEGMENT
        )
    for nick in intersect_nicks(filter, sup=False, eng_sup=False):
        yield from expand(
            RESULTS + "/" + nick + "/{test_corpus}-{test_seg}.db",
            test_corpus=CORPUS_NAMES,
            test_seg=TEST_SEGMENT
        )


def group_at_onces():
    filter = parse_filter(FILTER)
    for exp_group in SnakeMake.get_group_at_once_groups(filter.intersect_opts(sup=True)):
        # XXX: Is this a reasonable way to filter to a whole group?
        yield " ".join(exp_group.exps[0].path), exp_group


def group_at_once_nicks():
    filter = parse_filter(FILTER)
    return SnakeMake.get_group_at_once_nicks(filter.intersect_opts(sup=True))


def group_guesses(exp_group):
    return [
        GUESS + "/groupatonce/{train_corpus,[^/]+}-{train_seg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}/{group_nick,[^/]+}/" + exp.nick
        for exp in exp_group.exps
    ]


def get_corpus_seg(wc):
    return pjoin(CORPUS_DIR_MAP[wc.corpus](), wc.seg)


def get_sup_guess(wc):
    if wc.nick in group_at_once_nicks():
        return f"{GUESS}/groupatonce/{wc.train_corpusseg}/{wc.corpus}-{wc.seg}/{nick_to_group_nick_map[wc.nick]}/{wc.nick}"
    else:
        return f"{GUESS}/{wc.nick}/{wc.train_corpusseg}/{wc.corpus}-{wc.seg}"


## Top levels
rule all:
    input: list(all_results())

## Training

# Train supervised models
rule train_groupatonce:
    input: get_corpus_seg
    output: directory(WORK + "/models/groupatonce/{corpus,[^/]+}-{seg,[^/]+}/{group_nick,[^/]+}")
    run:
        shell(
            "mkdir -p {output} && " +
            "python scripts/expc.py --filter \"" + " ".join(path_nick_map[wildcards.group_nick]) + "\" train --multi " +
            "{input} {output}"
        )


rule train:
    input: get_corpus_seg
    output: directory(WORK + "/models/{corpus,[^/]+}-{seg,[^/]+}/{nick,[^/]+}")
    shell:
        "mkdir -p " + WORK + "/models/{wildcards.corpus}-{wildcards.seg}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" train {input} {output}"

rule train_eng:
    input: lambda wc: config["SEMCOR"]
    output: directory(WORK + "/models/semcor/{nick,[^/]+}")
    shell:
        "mkdir -p " + WORK + "/models/semcor/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" train {input} {output}"

## Testing

# Testing supervised models
rule test_sup_groupatonce:
    input:
        test = get_corpus_seg,
        model = WORK + "/models/groupatonce/{train_corpus}-{train_seg}/{group_nick}"
    output: directory(GUESS + "/groupatonce/{train_corpus,[^/]+}-{train_seg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}/{group_nick,[^/]+}")
    run:
        shell(
            "mkdir -p {output} && " +
            "python scripts/expc.py --filter \"" + " ".join(path_nick_map[wildcards.group_nick]) + "\" test --multi --model " +
            "{input.model} {input.test} {output}/__NICK__"
        )

rule test_sup:
    input:
        test = get_corpus_seg,
        model = WORK + "/models/{train_corpusseg}/{nick}",
    output: 
        GUESS + "/{nick,([^/](?!.x1st|.u1st))+}/{train_corpusseg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}"
    shell:
        "mkdir -p " + GUESS + "/{wildcards.nick}/{wildcards.train_corpusseg}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" test --model {input.model} {input.test} {output}"

rule test_sup_1st:
    input:
        test = get_corpus_seg,
        ceil_model = WORK + "/models/{train_corpusseg}/ceil.inst",
        guess_1st = GUESS + "/first/{corpus}-{seg}",
        inner_guess = GUESS + "/{inner_nick}/{train_corpusseg}/{corpus}-{seg}",
    output:
        GUESS + "/{inner_nick,[^/]+}.{type,(x1st|u1st)}/{train_corpusseg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}"
    shell:
        "mkdir -p " + GUESS + "/{wildcards.inner_nick}.{wildcards.type}/{wildcards.train_corpusseg}/ && "
        "GUESS_1ST={input.guess_1st} "
        "INNER_GUESS={input.inner_guess} "
        "CEIL_MODEL={input.ceil_model} "
        "python scripts/expc.py --filter \"nick={wildcards.inner_nick}.{wildcards.type}\" test --model {input.model} {input.test} {output}"

# Testing unsupervised models
rule test_unsup:
    input:
        test = get_corpus_seg,
    output:
        GUESS + "/{nick,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}"
    shell:
        "mkdir -p " + GUESS + "/{wildcards.nick}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" test {input.test} {output}"

## Fan out
for group_nick, exp_group in group_at_once_map.items():
    rule:
        input: GUESS + "/groupatonce/{train_corpus}-{train_seg}/{corpus}-{seg}/{group_nick}"
        output:
            group_guesses(exp_group),
            touch(GUESS + "/groupatonce/{train_corpus,[^/]+}-{train_seg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}/{group_nick,[^/]+}.fan-out")

## Scoring

# Scoring supervised models
rule eval_sup:
    input:
        test = get_corpus_seg,
        guess = get_sup_guess,
    output:
        RESULTS + "/{nick,[^/]+}/{train_corpusseg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}.db"
    shell:
        "mkdir -p " + RESULTS + "/{wildcards.nick}/{wildcards.train_corpusseg}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" eval {output} {input.guess} {input.test} train-corpus={wildcards.train_corpusseg} test-corpus={wildcards.corpus}-{wildcards.seg}"

# Scoring unsupervised models
rule eval_unsup:
    input:
        test = get_corpus_seg,
        guess = GUESS + "/{nick}/{corpus}-{seg}"
    output:
        RESULTS + "/{nick,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}.db"
    shell:
        "mkdir -p " + RESULTS + "/{wildcards.nick}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" eval {output} {input.guess} {input.test} test-corpus={wildcards.corpus}-{wildcards.seg}"

## Include bootstrapping tasks
include: "Snakefile.bs"
