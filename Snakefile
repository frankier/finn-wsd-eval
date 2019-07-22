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
cnf("STIFF_STIFF_GUESS", f"{WORK}/stiff-stiff-guess")
cnf("STIFF_EURO_GUESS", f"{WORK}/stiff-euro-guess")
cnf("EURO_STIFF_GUESS", f"{WORK}/euro-stiff-guess")
cnf("EURO_EURO_GUESS", f"{WORK}/euro-euro-guess")

#CORPUS_DIRS = [STIFFEVAL, EUROMODELS]
cnf_list("CORPUS_NAMES", ["stiff", "eurosense"])
cnf_list("TRAIN_SEGMENT", ["train", "test"])
cnf_list("TEST_SEGMENT", ["dev", "test"])

CORPUS_DIR_MAP = {
    "stiff": lambda: config["STIFFEVAL"],
    "eurosense": lambda: config["EUROSENSEEVAL"],
}

# Utility functions
def all_results():
    path, opt_dict = parse_filter(FILTER)
    for nick in SnakeMake.get_nicks(path, {"sup": True, **opt_dict}):
        yield from expand(
            WORK + "/results/" + nick + "/{train_corpus}-{train_seg}/{test_corpus}-{test_seg}.db",
            train_corpus=CORPUS_NAMES,
            train_seg=TRAIN_SEGMENT,
            test_corpus=CORPUS_NAMES,
            test_seg=TEST_SEGMENT
        )
    for nick in SnakeMake.get_nicks(path, {"sup": False, **opt_dict}):
        yield from expand(
            WORK + "/results/" + nick + "/{test_corpus}-{test_seg}.db",
            test_corpus=CORPUS_NAMES,
            test_seg=TEST_SEGMENT
        )

def get_corpus_seg(wc):
    return pjoin(CORPUS_DIR_MAP[wc.corpus](), wc.seg)

## Top levels
rule all:
    input: list(all_results())

## Training

# Train supervised models
rule train:
    input: get_corpus_seg
    output: directory(WORK + "/models/{corpus,[^/]+}-{seg,[^/]+}/{nick,[^/]+}")
    shell:
        "mkdir -p " + WORK + "/models/{wildcards.corpus}-{wildcards.seg}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" train {input} {output}"

## Testing

# Testing supervised models
rule test_sup:
    input:
        test = get_corpus_seg,
        model = WORK + "/models/{train_corpus}-{train_seg}/{nick}",
    output: 
        WORK + "/guess/{nick,[^/]+}/{train_corpus,[^/]+}-{train_seg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}"
    shell:
        "mkdir -p " + WORK + "/guess/{wildcards.nick}/{wildcards.train_corpus}-{wildcards.train_seg}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" test --model {input.model} {input.test} {output}"

# Testing unsupervised models
rule test_unsup:
    input:
        test = get_corpus_seg,
    output:
        WORK + "/guess/{nick,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}"
    shell:
        "mkdir -p " + WORK + "/guess/{wildcards.nick}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" test {input.test} {output}"

## Scoring

# Scoring supervised models
rule eval_sup:
    input:
        test = get_corpus_seg,
        guess = WORK + "/guess/{nick}/{train_corpus}-{train_seg}/{corpus}-{seg}"
    output:
        WORK + "/results/{nick,[^/]+}/{train_corpus,[^/]+}-{train_seg,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}.db"
    shell:
        "mkdir -p " + WORK + "/results/{wildcards.nick}/{wildcards.train_corpus}-{wildcards.train_seg}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" eval {output} {input.guess} {input.test} train-corpus={wildcards.train_corpus}-{wildcards.train_seg} test-corpus={wildcards.corpus}-{wildcards.seg}"

# Scoring unsupervised models
rule eval_unsup:
    input:
        test = get_corpus_seg,
        guess = WORK + "/guess/{nick}/{corpus}-{seg}"
    output:
        WORK + "/results/{nick,[^/]+}/{corpus,[^/]+}-{seg,[^/]+}.db"
    shell:
        "mkdir -p " + WORK + "/results/{wildcards.nick}/ && "
        "python scripts/expc.py --filter \"nick={wildcards.nick}\" eval {output} {input.guess} {input.test} test-corpus={wildcards.corpus}-{wildcards.seg}"
