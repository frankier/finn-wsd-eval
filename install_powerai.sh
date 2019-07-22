set -euxo pipefail

cpip() {
    ERROR_MSG="Not in a conda environment."
    ERROR_MSG="$ERROR_MSG\nUse \`source activate ENV\`"
    ERROR_MSG="$ERROR_MSG to enter a conda environment."

    [ -z "$CONDA_DEFAULT_ENV" ] && echo "$ERROR_MSG" && return 1

    ERROR_MSG='Pip not installed in current conda environment.'
    ERROR_MSG="$ERROR_MSG\nUse \`conda install pip\`"
    ERROR_MSG="$ERROR_MSG to install pip in current conda environment."

    [ -e "$CONDA_PREFIX/bin/pip" ] || (echo "$ERROR_MSG" && return 2)

    PIP="$CONDA_PREFIX/bin/pip"
    "$PIP" "$@"
}

# Add channels
conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/

## Install
conda install -y powerai
conda install -y pip
conda run cpip install poetry
conda run poetry export --without-hashes -f requirements.txt
# Obviously not compatible
sed -i '/pypiwin32/d' requirements.txt
sed -i '/pywin32/d' requirements.txt
# Already installed
sed -i '/sklearn/d' requirements.txt
sed -i '/torch/d' requirements.txt
conda run cpip install --pre -r requirements.txt

## Init
conda run python -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"
conda run python -m finntk.scripts.bootstrap_all
conda run python fetchers/ukb.py fetch
bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && conda run python fetchers/supwsd.py'
conda run python fetchers/ctx2vec.py
conda run python fetchers/sif.py
conda run python -m stiff.scripts.post_install
