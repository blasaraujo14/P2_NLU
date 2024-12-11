from conllu_token import Token
from conllu_reader import ConlluReader
from algorithm import ArcEager, Sample, Transition
import numpy as np
import tensorflow as tf
import keras
from keras import Model
from keras.layers import TextVectorization
from keras.models import Sequential
from keras import layers
from keras.layers import Input, Dense, Embedding, TimeDistributed, LSTM

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 64, 
                 epochs: int = 1, batch_size: int = 64):
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.epoch = epochs
        self.batch_size = batch_size
        self.lr = 0.0005
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        #raise NotImplementedError

    def setLearningRate(self, lr):
        self.lr = lr

    def buildEncoding(self, data, init):
        encoding = {}
        code = init
        for text in data:
            if text not in encoding:
                encoding[text] = code # for encoding
                encoding[code] = text # for decoding
                code += 1
        return encoding

    def buildTargets(self, targets):
        actions = np.array([self.actionEncoding[t[0]] for t in targets])
        deps = np.array([self.depEncoding.get(t[1], 0) for t in targets])
        return (tf.keras.utils.to_categorical(actions), tf.keras.utils.to_categorical(deps))

    def buildFeatures(self, feats):
        return tf.convert_to_tensor([[self.featsEncoding.get(feat, 0) for feat in features] for features in feats])
    
    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """
        nbuffer_feats = 2; nstack_feats = 2

        #extracting the data from the samples for features and targets
        featsTrain = np.array([sample.state_to_feats(nbuffer_feats, nstack_feats) for sample in training_samples])
        targetsTrain = np.array([[sample.transition.action, sample.transition.dependency] for sample in training_samples])

        featsDev = np.array([sample.state_to_feats(nbuffer_feats, nstack_feats) for sample in dev_samples])
        targetsDev = np.array([[sample.transition.action, sample.transition.dependency] for sample in dev_samples])

        # encoding the many different string labels in integers
        codeActions = [sentence[0] for sentence in targetsTrain]
        codeDeps = [sentence[1] for sentence in targetsTrain]
        self.actionEncoding = self.buildEncoding(codeActions, 0)
        self.depEncoding = self.buildEncoding(codeDeps, 1)
        self.featsEncoding = self.buildEncoding(np.reshape(featsTrain, -1), 1)
        vocab_size = int(len(self.featsEncoding)/2) + 1

        targetsTrain = self.buildTargets(targetsTrain)
        targetsDev = self.buildTargets(targetsDev)

        featsTrain = self.buildFeatures(featsTrain)
        featsDev = self.buildFeatures(featsDev)

        # building the model architecture
        inputs = keras.Input(shape=(2 * (nbuffer_feats + nstack_feats),),dtype=tf.int32)

        x = layers.Embedding(input_dim=vocab_size, output_dim=self.word_emb_dim)(inputs)
        x = layers.Flatten()(x)

        xAction = layers.Dense(self.hidden_dim, activation='sigmoid')(x)
        xDep = layers.Dense(self.hidden_dim, activation='sigmoid')(x)

        outputsAction = layers.Dense(int(len(self.actionEncoding)/2), activation='softmax')(xAction)
        outputsDep = layers.Dense(int(len(self.depEncoding)/2+1), activation='softmax')(xDep)
        self.model = keras.Model(inputs=inputs, outputs=(outputsAction, outputsDep))

        self.model.summary()

        # model fitting
        # try needed for compatibility, earlier Tensorflow versions require different metrics argument
        try:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            self.model.fit(featsTrain, targetsTrain, batch_size=self.batch_size, epochs=self.epoch, validation_data=(featsDev, targetsDev))
        except: # if tensorflow is too new
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'accuracy']
            )

            self.model.fit(featsTrain, targetsTrain, batch_size=self.batch_size, epochs=self.epoch, validation_data=(featsDev, targetsDev))


        #raise NotImplementedError

    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
        # string label data is retrieved and encoded into integers
        inputs = np.array([sample.state_to_feats() for sample in samples])
        targets = np.array([[sample.transition.action, sample.transition.dependency] for sample in samples])
        targets = self.buildTargets(targets)
        inputs = self.buildFeatures(inputs)

        # we evaluate the model once
        self.model.evaluate(inputs, targets)
        #raise NotImplementedError
    
    def run(self, sents: list['Token']):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                 as a list of Token objects.
        """
        arc_eager = ArcEager()

        # origIndex will provide the original index of a sentence given its state representation
        origIndex = dict()
        def createStateAndIndex(i):
            state = arc_eager.create_initial_state(sents[i])
            origIndex[state] = i
            return state

        states = [createStateAndIndex(i) for i in range(len(sents))]

        # function to convert categorical model output to original string labels
        def decodeTargets(targets, encoding):
            # from categorical to original encoding
            originalEncoding = tf.argmax(targets, axis=1)
            def fn(target):
                out = encoding.get(target.numpy(), "unk")
                if out is None:
                    return 'None'
                return out

            return tf.map_fn(fn=fn,
                        elems=originalEncoding,
                        fn_output_signature=tf.string)
        
        # function that discards impossible actions and picks most probable ones
        def selectActions(sortedActions, index):
            action = ''
            for i in sortedActions[::-1]:
                action = self.actionEncoding[i.numpy()]
                if action == arc_eager.LA:
                    if arc_eager.LA_is_valid(states[index]):
                        break
                elif action == arc_eager.RA:
                    if arc_eager.RA_is_valid(states[index]):
                        break
                elif action == arc_eager.REDUCE:
                    if arc_eager.REDUCE_is_valid(states[index]):
                        break
                else:
                    break
            return action

        # at the end of execution will hold the list of model predicted transitions for each sentence
        transLists = [[] for i in range(len(sents))]
        # applies transition to state and appends transition to list corresponding to a sentence
        def registerTransition(trans, state):
            arc_eager.apply_transition(state, trans)
            transLists[origIndex[state]].append(trans)

        cnt = 1
        while(len(states)!=0):
            #print("Iteration " + str(cnt)); cnt += 1
            # get string label features for each sample and encoding them in integers
            feats = np.array([Sample(state, None).state_to_feats() for state in states])
            feats = self.buildFeatures(feats)

            # running inference and sorting actions by probability
            actions, deps = self.model(feats)
            sorted_actions = np.argsort(actions, axis=-1)
            #print("Sorted indexes: " + str(sorted_actions))

            # action selection/discarding
            selected_actions = tf.map_fn(
                fn=lambda x: selectActions(x[0], x[1]),
                elems=(sorted_actions, tf.range(len(states))),
                fn_output_signature=tf.string)

            # decoding categorical model output
            deps = list(decodeTargets(deps, self.depEncoding).numpy())
            deps = [dep.decode('utf-8') for dep in deps]

            # building a transition with said output and registering it for that state's sentence
            [registerTransition(Transition(selected_actions[i], deps[i]), states[i]) for i in range(len(states))]

            # remove final states
            states = [state for state in states if not arc_eager.final_state(state)]

        # function that takes a sentence of tokens and a list transitions, and applied the dependencies to it
        def toCoNLLU(sent: list['Token'], trans: list['Transition']) -> list['Token']:
            state = arc_eager.create_initial_state(sent)
            currentArcs = set()

            # applies the transitions
            for transition in trans:

                arc_eager.apply_transition(state, transition)

                # determine if a new arc has been created
                arcDiff = state.A - currentArcs
                if arcDiff != set():
                    # only one new arc can result from applying a transition
                    head_id, dependency_label, dependent_id = arcDiff.pop()

                    token = sent[dependent_id]
                    assert token.id == dependent_id, f"Token ID mismatch: {token.id} != {dependent_id}."

                    token.head = head_id
                    token.dep = dependency_label
                    currentArcs.add((head_id, dependency_label, dependent_id))

            return sent

        # return sentences with dependency information from model predictions
        return [toCoNLLU(sents[i], transLists[i]) for i in range(len(sents))]

        # Main Steps for Processing Sentences:
        # 1. Initialize: Create the initial state for each sentence.
        # 2. Feature Representation: Convert states to their corresponding list of features.
        # 3. Model Prediction: Use the model to predict the next transition and dependency type for all current states.
        # 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
        #    and select the most likely dependency type with argmax.
        # 5. Validation Check: Verify if the selected transition is valid for each prediction. If not, select the next most likely one.
        # 6. State Update: Apply the selected actions to update all states, and create a list of new states.
        # 7. Final State Check: Remove sentences that have reached a final state.
        # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.


        #raise NotImplementedError


if __name__ == "__main__":
    def read_file(reader, path, inference):
        trees = reader.read_conllu_file(path, inference)
        #print(f"Read a total of {len(trees)} sentences from {path}")
        #print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
        #for token in trees[0]:
            #print (token)e
        #print ()
        return trees
    train = True
    model = ParserMLP(epochs=100)
    if train:
        reader = ConlluReader()
        arc_eager = ArcEager()

        train_trees = read_file(reader,path="en_partut-ud-train_clean.conllu", inference=False)
        dev_trees = read_file(reader,path="en_partut-ud-dev_clean.conllu", inference=False)
        test_trees = read_file(reader,path="en_partut-ud-test_clean.conllu", inference=True)
        """
        We remove the non-projective sentences from the training and development set,
        as the Arc-Eager algorithm cannot parse non-projective sentences.

        We don't remove them from test set set, because for those we only will do inference
        """
        train_trees = reader.remove_non_projective_trees(train_trees)[:100]
        dev_trees = reader.remove_non_projective_trees(dev_trees)[:50]

        samplesT = np.concatenate([arc_eager.oracle(tree) for tree in train_trees], 0)
        samplesD = np.concatenate([arc_eager.oracle(tree) for tree in dev_trees], 0)

        model.train(samplesT, samplesD)
    model.evaluate(samplesD)
    print(model.run(test_trees))
    
