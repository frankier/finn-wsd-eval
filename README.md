# Evaluation framework

<p align="center">
<a href="https://gitlab.com/frankier/finn-wsd-eval/pipelines"><img alt="pipeline status" src="https://gitlab.com/frankier/finn-wsd-eval/badges/master/pipeline.svg" /></a>
</p>

## Setup

The evaluation framework is distributed together with all systems and
requirements as a Docker image, which can be pulled like so:

    $ docker pull registry.gitlab.com/frankier/stiff:latest

And run like so:

    $ docker -v /path/to/working/dir/:/work/ run python eval.py /work/results.json /work/eurosense.eval/

For CUDA accelerated experiments, you can use
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). For running in shared
computing environments, in which you don't have root access, I recommend
[udocker](https://github.com/indigo-dc/udocker).

You can also set up requirements manually. See the `Dockerfile` for the list of
commands to run.

## Evaluation corpus

The one requirement which is not included in the Docker image is the evaluation
corpus. Please follow [the instructions in the STIFF
README](https://github.com/frankier/STIFF#eurosense-pipeline).

## Filtering experiments with `eval.py`

You can run a subset of experiments by passing filters to `eval.py`, e.g.

    $ python eval.py /work/results.json /work/eurosense.eval/ Knowledge 'Cross-lingual Lesk' mean=pre_sif_mean

## Making tables with `table.py`

You can make LaTeX tables with `table.py`, e.g.

    $ python table.py results.json --filter='Knowledge;Cross-lingual Lesk' --table='use_freq;vec:fasttext,numberbatch,double;mean expand;wn_filter'

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
