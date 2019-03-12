import sys
import pickle
from context2vec.eval.wsd.dataset_reader import DatasetReader
from context2vec.common.model_reader import ModelReader

k = 1
isolate_target_sentence = True
ignore_closest = False

test_filename = sys.argv[1]
model_params_filename = sys.argv[2]
model_in = sys.argv[3]
result_filename = sys.argv[4]

knn = pickle.load(open(model_in, "rb"))
print("Reading model..")
model_reader = ModelReader(model_params_filename)
model = model_reader.model
dataset_reader = DatasetReader(model)

print("Reading test dataset..")
test_set, test_key2ind, test_ind2key = dataset_reader.read_dataset(
    test_filename, test_filename + ".key", False, isolate_target_sentence
)

print("Starting to classify test set:")
with open(result_filename, "w") as o:
    for ind, key_set in enumerate(test_set):
        key = test_ind2key[ind]
        for instance_id, vec, text in zip(
            key_set.instance_ids, key_set.context_m, key_set.contexts_str
        ):
            result = knn.classify(key, vec, ignore_closest, False)
            # brother.n 00006 501566/0.5 501573/0.4 503751/0.1
            result_line = key + " " + instance_id
            for sid, weight in result.items():
                result_line += " {}/{:.4f}".format(sid, weight)

            o.write(result_line + "\n")
