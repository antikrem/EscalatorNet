# Provdes interface for generating escalator networks

from e_net_engine import Network_create, Network_delete, Network_setHyperParameter, Network_get, Network_addExamples, Network_train, Network_predict, version
from enumerations import FunctionTypes, Parameters

class Network :
    '''
    Represents a full network
    '''

    def __init__(
                self, 
                nodeCount,
                functiontype = FunctionTypes.sigmoid,
                learningrate = 1.0
            ) :
        '''
        Creates a Network with given node counts
        The first count will be the size of input of the input layer
        '''
        self._netPtr = Network_create(nodeCount, functiontype.value)
        self.version = version()
        self.set_parameter(Parameters.learning_rate, learningrate)

    def __del__(self):
        '''
        Destroys underlying network
        '''
        Network_delete(self._netPtr)

    def set_parameter(self, parameter, value) :
        '''
        Updates a hyper parameter in this model
        Takes a Parameter as an argument
        '''
        Network_setHyperParameter(self._netPtr, parameter.value, value)

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

    def predict(self, arg1, arg2 = None) :
        '''
        Make a prediction with given data
        Has 2 overloads, whether it has 1 or multiple rows:

        Input 1 (single row representing one input prediction):
            arg1: Single row of data for prediction
            arg2: Leave empty

        Input 2 (single row representing multuple input prediction)
            arg1: Number of inputs
            arg2: Single row with every prediction input concatenated
        '''
        if arg2 is None :
            return Network_predict(self._netPtr, 1, arg1)
        else :
            return Network_predict(self._netPtr, arg1, arg2)
        