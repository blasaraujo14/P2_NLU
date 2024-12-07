
from conllu_reader import ConlluReader
from algorithm import ArcEager
from model import ParserMLP
from postprocessor import PostProcessor
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

#Create an instance of the ArcEager
arc_eager = ArcEager()

# Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.
# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.

# Implement the 'state_to_feats' function in the Sample class.
# This function should convert the current parser state into a list of features for use by the neural model classifier.

print("Getting the sample datasets.")
build_datasets = True

if build_datasets:
    samplesT = np.concatenate([arc_eager.oracle(tree) for tree in train_trees], 0)
    samplesD = np.concatenate([arc_eager.oracle(tree) for tree in dev_trees], 0)

    np.save("samplesTrain.npy", samplesT)
    np.save("samplesDev.npy", samplesD)
else:
    samplesT, samplesD = np.load("samplesTrain.npy", allow_pickle=True), np.load("samplesDev.npy", allow_pickle=True)

# Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training dataset.
# 2. Evaluate the model's performance using the development dataset.
# 3. Conduct inference on the test set with the trained model.
# 4. Save the parsing results of the test set in CoNLLU format for further analysis.

print("Starting inference process.")
make_inferences = True
corruptedPath = "corrupted_inferences.conllu"

if make_inferences:
    model = ParserMLP(epochs=10)

    model.train(samplesT, samplesD)

    print("Training finished. Final validation dataset evaluation:")
    model.evaluate(samplesD)

    print("Running test dataset inference vertically, this may take a while.")
    # make inferences and save them in CoNLLU format.
    inferences = model.run(test_trees)
    reader.write_conllu_file(corruptedPath, inferences)

# Utilize the 'postprocessor' module (already implemented).
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
# 3. Process the file: trees = postprocessor.postprocess(path)
# 4. Save the processed trees to a new output file.

print("Post-processing potentially corrupted inferences...")
inferences = reader.read_conllu_file(corruptedPath, inference=False)

p = PostProcessor()
trees = p.postprocess(corruptedPath)
reader.write_conllu_file("inferences.conllu", trees)

print("Readdy for evaluation with command:")
print("python conll18_ud_eval.py en_partut-ud-test_clean.conllu inferences.conllu -v")
