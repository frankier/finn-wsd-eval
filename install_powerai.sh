set -euxo pipefail

PYTHON="$CONDA_PREFIX/bin/python"

cpip() {
    ERROR_MSG="Not in a conda environment."
    ERROR_MSG="$ERROR_MSG\nUse \`source activate ENV\`"
    ERROR_MSG="$ERROR_MSG to enter a conda environment."

    [ -z "$CONDA_DEFAULT_ENV" ] && echo "$ERROR_MSG" && return 1

    ERROR_MSG='Pip not installed in current conda environment.'
    ERROR_MSG="$ERROR_MSG\nUse \`conda install pip\`"
    ERROR_MSG="$ERROR_MSG to install pip in current conda environment."

    [ -e "$CONDA_PREFIX/bin/pip" ] || (echo "$ERROR_MSG" && return 2)

    mkdir -p "$CONDA_PREFIX/src"
    PIP="$CONDA_PREFIX/bin/pip"
    PIP_SRC="$CONDA_PREFIX/src" "$PIP" "$@"
}

# Add channels
conda config --prepend channels conda-forge
conda config --prepend channels bioconda
conda config --prepend channels defaults
conda config --prepend channels https://oplab9.parqtec.unicamp.br/pub/ppc64el/power-ai/linux-ppc64le/
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

## Install
conda install -y pip

# Binary packages that will be filtered out of requirements
conda install -y graphviz
conda install -y pygraphviz
conda install -y powerai
conda install -y cupy
conda install -y chainer
conda install -y gensim
conda install -y -c anaconda datrie

cpip install --pre poetry
conda run poetry export --without-hashes -f requirements.txt

# Obviously not compatible
sed -i '/pypiwin32/d' requirements.txt
sed -i '/pywin32/d' requirements.txt
# Already installed
sed -i '/sklearn/d' requirements.txt
sed -i '/^torch/d' requirements.txt
sed -i '/scikit-learn/d' requirements.txt
sed -i '/scipy/d' requirements.txt
sed -i '/numpy/d' requirements.txt
sed -i '/gensim/d' requirements.txt
sed -i '/h5py/d' requirements.txt
sed -i '/datrie/d' requirements.txt
sed -i '/pygraphviz/d' requirements.txt
# Apparently not compatible
sed -i '/hfst/d' requirements.txt
sed -i '/wrapt/d' requirements.txt
# These need to be installed in non-editable mode since they're pyproject.toml based
sed -i '/STIFF/d' requirements.txt
sed -ie '/expcomb/s/^-e //' requirements.txt
sed -ie '/memory-tempfile/s/^-e //' requirements.txt

rm -rf "$CONDA_PREFIX/src/" || true
cpip install --no-deps --pre -r requirements.txt

# STIFF is very resistant to being installed normally - manual install
git clone -q https://github.com/frankier/STIFF.git $CONDA_PREFIX/src/STIFF
echo "$CONDA_PREFIX/src/STIFF" > $CONDA_PREFIX/lib/python3.6/site-packages/stiff.pth

# Finally install wsdeval
echo "`pwd`" > $CONDA_PREFIX/lib/python3.6/site-packages/wsdeval.pth

## Init
$PYTHON -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"
$PYTHON -m finntk.scripts.bootstrap_all
$PYTHON -m stiff.scripts.post_install
$PYTHON fetchers/ctx2vec.py --skip-pip
# Fetch BERT/ELMo
$PYTHON -c "from finntk.emb.bert import vecs; vecs.get()"
$PYTHON -c "from finntk.emb.elmo import vecs; vecs.bootstrap()"

# Now context2vec has to be install manually too
echo "`pwd`/systems/context2vec/" > $CONDA_PREFIX/lib/python3.6/site-packages/context2vec.pth
