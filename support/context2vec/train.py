import pickle
import sys
from context2vec.eval.wsd.knn import Knn
from context2vec.eval.wsd.dataset_reader import DatasetReader
from context2vec.common.model_reader import ModelReader

k = 1
isolate_target_sentence = True
ignore_closest = False

train_filename = sys.argv[1]
model_params_filename = sys.argv[2]
model_out = sys.argv[3]

print("Reading model..")
model_reader = ModelReader(model_params_filename)
model = model_reader.model
dataset_reader = DatasetReader(model)

print("Reading train dataset..")
train_set, train_key2ind, train_ind2key = dataset_reader.read_dataset(
    train_filename, train_filename + ".key", True, isolate_target_sentence
)
knn = Knn(k, train_set, train_key2ind)
pickle.dump(knn, open(model_out, "wb"))
