# Provdes interface for generating escalator networks

from e_net_engine import Network_create, Network_delete, Network_get, Network_addExamples, Network_train, Network_predict, version

class Network :
    '''
    Represents a full network
    '''

    def __init__(self, nodeCount) :
        '''
        Creates a Network with given node counts
        The first count will be the size of input of the input layer
        '''
        self._netPtr = Network_create(nodeCount)
        self.version = version()

    def __del__(self):
        '''
        Destroys underlying network
        '''
        Network_delete(self._netPtr)

    def _validate(self) :
        print(Network_get(self._netPtr))

    def add_example(self, input, output) :
        '''
        Takes a single row of training data
        '''
        Network_addExamples(self._netPtr, 1, input, output)

    def add_examples(self, count, input, output) :
        '''
        Takes multiple examples for testing
        '''
        Network_addExamples(self._netPtr, count, input, output)

    def train(self) :
        '''
        Trains the network with the internalally set trainging data
        '''
        Network_train(self._netPtr)

    def predict(self, input) :
        '''
        Make a prediction with given data
        '''
        return Network_predict(self._netPtr, 1, input)