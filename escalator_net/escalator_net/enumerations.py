from enum import Enum

class FunctionTypes(Enum) :
    sigmoid = "sigmoid"
    ReLU = "ReLU"
    LeakyReLU = "LeakyReLU"
    softplus = "softplus"

class Parameters(Enum) :
    convergence_threshold = "convergence_threshold"
    iteration_max = "iteration_max"
    learning_rate = "learning_rate"