# Evaluation framework

<p align="center">
<a href="https://gitlab.com/frankier/finn-wsd-eval/pipelines"><img alt="pipeline status" src="https://gitlab.com/frankier/finn-wsd-eval/badges/master/pipeline.svg" /></a>
</p>

This directory contains the evaluation framework. The scorer program is the
same as the one used in *Word Sense Disambiguation: A Unified Evaluation
Framework and Empirical Comparison.*. First you need to obtain it:

    ./get_scorer.sh

## Running the baselines

    ./run_eval_baselines.sh /path/to/eval.xml /path/to/eval.key

## Running the UKB experiments

First set the environment variable UKB_PATH to where your compiled copy of UKB
is located.

    cd ukb-eval && ./prepare_wn30graph.sh && cd ..
    pipenv run python mkwndict.py --en-synset-ids > wndict.en.txt
    pipenv run python ukb.py run_all /path/to/eval/corpus.xml ukb-eval/wn30/wn30g.bin wndict.txt /path/to/eval/corpus.key

## Licenses ##

This project is licensed under the Apache v2 license. The code in `ukb-eval` is
vendorized from UKB, and therefore licensed under the GPL. The scorer in
`support/scorer` is under an unknown license, possibly public domain.

## See also

 * [STIFF](https://github.com/frankier/STIFF): Automatically created sense
   tagged corpus of Finnish and corpus wrangling tools.
 * [STIFF-explore](https://github.com/frankier/STIFF-explore): Some exploratory
   coding related to STIFF.
 * [finn-man-ann](https://github.com/frankier/finn-man-ann): Small, Finnish
   language, manually annotated word sense corpus.
 * [FinnTK](https://github.com/frankier/finntk): Simple, high-level toolkit for
   Finnish NLP, mainly providing convenience methods for, and gluing together,
   other tools.
 * [extjwnl_fiwn](https://github.com/frankier/extjwnl_fiwn): Java code to make
   extjwnl interoperate with FinnWordNet.
 * [FinnLink](https://github.com/frankier/FinnLink): Link between FinnWordNet
   and Finnish Propbank created by joining with PredicateMatrix.
 * [finn-sense-clust](https://github.com/frankier/finn-sense-clust): Sense
   clusterings of FinnWordNet.

### Forks/fixes

 * [ItMakeseSense](https://github.com/frankier/ims): ItMakesSense fork to
   support FiWN for use by finn-wsd-eval
 * [AutoExtend](https://github.com/frankier/AutoExtend): AutoExtend fork to
   support FiWN and ConceptNet Numberbatch
 * [babelnet-lookup](https://github.com/frankier/babelnet-lookup):
   babelnet-lookup fork to obtain `BABEL2WN_MAP`.
 * [FinnWordNet](https://github.com/frankier/fiwn): Temporary fixes to
   FinnWordNet 2.0.
