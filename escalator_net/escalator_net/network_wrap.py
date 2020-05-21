# Provdes interface for generating escalator networks

from e_net_engine import Network_create, Network_delete, Network_get, Network_addExample, Network_train, version

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
        Network_addExample(self._netPtr, input, output)

    def train(self) :
        '''
        Takes a single row of training data
        '''
        Network_train(self._netPtr)

    def predict(self) :
        '''
        Takes a single row of input 
        '''
        pass