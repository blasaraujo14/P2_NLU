
from conllu_reader import ConlluReader
from algorithm import ArcEager
import numpy as np

def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees


"""
ALREADY IMPLEMENTED
Read and convert CoNLLU files into tree structures
"""
# Initialize the ConlluReader
reader = ConlluReader()
train_trees = read_file(reader,path="en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="en_partut-ud-test_clean.conllu", inference=True)

"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""
train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))

#Create and instance of the ArcEager
arc_eager = ArcEager()

print ("\n ------ TODO: Implement the rest of the assignment ------")

# TODO: Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.

# TODO: Implement the 'state_to_feats' function in the Sample class.
# This function should convert the current parser state into a list of features for use by the neural model classifier.

build_datasets = False

def getFeatsTargets(trees):
    samples = np.concatenate([arc_eager.oracle(tree) for tree in trees], 0)
    feats = np.array([sample.state_to_feats() for sample in samples])
    targets = np.array([[sample.transition.action, sample.transition.dependency] for sample in samples])

    return feats, targets

for sample in arc_eager.oracle(train_trees[1]):
    print(sample.state, end='')
    print(sample.state_to_feats())
    print(sample.transition)
    print()

if build_datasets:
    feats, targets = getFeatsTargets(train_trees)

    np.save("inputsTrain.npy", feats)
    np.save("targetsTrain.npy", targets)

    feats, targets = getFeatsTargets(dev_trees)

    np.save("inputsVal.npy", feats)
    np.save("targetsVal.npy", targets)

    #feats, targets = getFeatsTargets(test_trees)

    #np.save("inputsTest.npy", feats)
    #np.save("targetsTest.npy", targets)

# TODO: Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training dataset.
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.

# TODO: Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.
