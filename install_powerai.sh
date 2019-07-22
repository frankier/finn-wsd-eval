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
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

## Install
conda install -y powerai
conda install -y pip
cpip install --pre poetry
conda run poetry export --without-hashes -f requirements.txt

# Obviously not compatible
sed -i '/pypiwin32/d' requirements.txt
sed -i '/pywin32/d' requirements.txt
# Already installed
sed -i '/sklearn/d' requirements.txt
sed -i '/torch/d' requirements.txt
sed -i '/scikit-learn/d' requirements.txt
sed -i '/scipy/d' requirements.txt
sed -i '/numpy/d' requirements.txt
sed -i '/gensim/d' requirements.txt
sed -i '/h5py/d' requirements.txt
# Apparently not compatible
sed -i '/hfst/d' requirements.txt
sed -i '/STIFF/d' requirements.txt
sed -ie '/expcomb/s/^-e //' requirements.txt
sed -i '/wrapt/d' requirements.txt

rm -rf "$CONDA_PREFIX/src/" || true
cpip install --no-deps --pre -r requirements.txt

## Init
$PYTHON -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"
$PYTHON -m finntk.scripts.bootstrap_all
$PYTHON fetchers/ukb.py fetch
bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && conda run $PYTHON fetchers/supwsd.py'
$PYTHON fetchers/ctx2vec.py
$PYTHON fetchers/sif.py
$PYTHON -m stiff.scripts.post_install
