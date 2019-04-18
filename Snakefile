## Environment variables
from os.path import join as pjoin

def cnf(name, val):
    globals()[name] = config.setdefault(name, val)

# Arguments
cnf("FILTER", "")

# Input corpora (required)
STIFFEVAL = config["STIFFEVAL"]
EUROSENSEEVAL = config["STIFFEVAL"]

# Intermediate dirs
cnf("WORK", "work")
cnf("STIFF_STIFF_GUESS", f"{WORK}/stiff-stiff-guess")
cnf("STIFF_EURO_GUESS", f"{WORK}/stiff-euro-guess")
cnf("EURO_STIFF_GUESS", f"{WORK}/euro-stiff-guess")
cnf("EURO_EURO_GUESS", f"{WORK}/euro-euro-guess")

#CORPUS_DIRS = [STIFFEVAL, EUROMODELS]
CORPUS_NAMES = ["stiff", "eurosense"]
CORPUS_DIR_MAP = {
    "stiff": STIFFEVAL,
    "eurosense": EUROSENSEEVAL,
}
TRAIN_SEGMENT = ["train", "test"]  # 
TEST_SEGMENT = ["dev", "test"]

# Result db
cnf("DB", "db.json")

# XXX TODO: Separate out unsupervised experiments that don't depend on training data

## Top levels
rule std:
    input: "std_sup.done"

rule std_sup:
    input:
        expand(WORK + "/guess/{train_corpus}-train-{test_corpus}-test",
               train_corpus=CORPUS_NAMES, test_corpus=CORPUS_NAMES)
    output: touch("std_sup.done")

rule self_test_sup:
    input:
        expand(WORK + "/guess/{train_corpus}-test-{test_corpus}-test",
               train_corpus=CORPUS_NAMES, test_corpus=CORPUS_NAMES)

## Training

# Train supervised models
rule train:
    input: lambda wc: pjoin(CORPUS_DIR_MAP[wc.corpus], wc.seg)
    output: WORK + "/models/{corpus}-{seg}"
    shell:
        "python scripts/eval.py --filter \"{FILTER}\" train {input} {output}"

## Evaluations

# Evaluate supervised models
rule eval_sup:
    input:
        model = WORK + "/models/{train_corpus}-{train_seg}",
        test = lambda wc: pjoin(CORPUS_DIR_MAP[wc.test_corpus], wc.test_seg)
    output: 
        WORK + "/guess/{train_corpus}-{train_seg}-{test_corpus}-{test_seg}"
    shell: "python scripts/eval.py --filter \"{FILTER}\" test {input.test} {DB} {output} {input.model}"
