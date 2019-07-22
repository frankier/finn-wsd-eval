set -euxo pipefail

## Install
conda install powerai
conda install pip
conda run pip install poetry
conda run poetry export --without-hashes -f requirements.txt
# Obviously not compatible
sed -i '/pypiwin32/d' requirements.txt
sed -i '/pywin32/d' requirements.txt
# Already installed
sed -i '/sklearn/d' requirements.txt
sed -i '/torch/d' requirements.txt
conda run pip install --pre -r requirements.txt

## Init
conda run python -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"
conda run python -m finntk.scripts.bootstrap_all
conda run python fetchers/ukb.py fetch
bash -c 'source "/root/.sdkman/bin/sdkman-init.sh" && conda run python fetchers/supwsd.py'
conda run python fetchers/ctx2vec.py
conda run python fetchers/sif.py
conda run python -m stiff.scripts.post_install
