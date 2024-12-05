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
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        #raise NotImplementedError

    def buildEncoding(data, init):
        encoding = {}
        code = init
        for text in data:
            if text not in encoding:
                encoding[text] = code # for encoding
                encoding[code] = text # for decoding
                code += 1
        return encoding

    def buildTargets(targets, actionEncoding, depEncoding):
        actions = tf.convert_to_tensor([actionEncoding[t[0]] for t in targets])
        dep = tf.convert_to_tensor([depEncoding.get(t[1], 0) for t in targets])
        targets = tf.concat([tf.keras.utils.to_categorical(actions), tf.keras.utils.to_categorical(dep)], axis=-1)
        return targets
    
    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """

        featsTrain = tf.convert_to_tensor([' '.join(sample.state_to_feats()) for sample in training_samples])
        targetsTrain = np.array([[sample.transition.action, sample.transition.dependency] for sample in training_samples])

        featsDev = tf.convert_to_tensor([' '.join(sample.state_to_feats()) for sample in dev_samples])
        targetsDev = np.array([[sample.transition.action, sample.transition.dependency] for sample in dev_samples])

        codeActions = [sentence[0] for sentence in targetsTrain]
        codeDeps = [sentence[1] for sentence in targetsTrain]
        actionEncoding = ParserMLP.buildEncoding(codeActions, 0)
        depEncoding = ParserMLP.buildEncoding(codeDeps, 1)

        targetsTrain = ParserMLP.buildTargets(targetsTrain, actionEncoding, depEncoding)
        targetsDev = ParserMLP.buildTargets(targetsDev, actionEncoding, depEncoding)

        #text_vectorizer = layers.TextVectorization(output_sequence_length=100)
        text_vectorizer = layers.TextVectorization(output_mode='int', output_sequence_length=8)
        text_vectorizer.adapt(featsTrain)

        inputs = keras.Input(shape=(1,),dtype=tf.string)
        x = text_vectorizer(inputs)
        vocab_size = text_vectorizer.vocabulary_size()

        x = layers.Embedding(input_dim=vocab_size, output_dim=self.word_emb_dim)(x)
        x = layers.Dense(self.hidden_dim, activation='sigmoid')(x)
        x = layers.GlobalMaxPooling1D()(x)
        outputs = layers.Dense(len(actionEncoding)/2+len(depEncoding)/2+1, activation='softmax')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.summary()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(featsTrain[0])

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
        inputs = tf.convert_to_tensor([' '.join(sample.state_to_feats()) for sample in samples])
        targets = np.array([[sample.transition.action, sample.transition.dependency] for sample in samples])
        codeActions = [sentence[0] for sentence in targets]
        codeDeps = [sentence[1] for sentence in targets]
        actionEncoding = ParserMLP.buildEncoding(codeActions, 0)
        depEncoding = ParserMLP.buildEncoding(codeDeps, 1)

        targets = ParserMLP.buildTargets(targets, actionEncoding, depEncoding)
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
        '''
        def getTransition(output):
            action = output[-4:]
            print (action)
            dep = output[:-4]
            print(dep)
            #transition = Transition(action, dep)
            return 1
        arc_eager = ArcEager()
        initial_state = arc_eager.create_initial_state(sents)
        feats = tf.convert_to_tensor([' '.join(Sample(initial_state, transition=None).state_to_feats())])
        #feats = tf.convert_to_tensor([Sample(initial_state, transition=None).state_to_feats()])
        output = self.model(feats)
        transition = getTransition(output[0])
        '''

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


        raise NotImplementedError


if __name__ == "__main__":
    def read_file(reader, path, inference):
        trees = reader.read_conllu_file(path, inference)
        #print(f"Read a total of {len(trees)} sentences from {path}")
        #print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
        #for token in trees[0]:
            #print (token)
        #print ()
        return trees
    
    model = ParserMLP(epochs=10)
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
    #samplesTest = np.concatenate([arc_eager.oracle(tree) for tree in test_trees], 0)

    model.train(samplesT, samplesD)
    model.run(train_trees[0])