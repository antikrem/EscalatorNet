/* Single node for a neural network
*/
#ifndef __NODE__
#define __NODE__

#include "functions.hpp"
#include "rand_ex.hpp"

/* Node in a neural network
 * templated to support different variable types
 */
template <typename T>
class Node {
	typename Functions<T>::function_ptr activationFunction = nullptr;

	// Threshold for a node activation
	const T activationThreshold = T(0.5);

	// size of input
	uint inputSize;

	// Associated weights
	VMatrix<T> weight;

	// Bias for node
	T bias = 0;

	// Randomize weights
	void randomiseWeights() {
		rand_ex::sampleNextUniforms(weight.get(), inputSize, 0.0, 1.0);
	}

public:
	/* Takes activation function as parameter
	 */
	Node(typename Functions<T>::function_ptr activationFunction, int inputSize)
	: inputSize(inputSize), weight(1, inputSize, 0) {
		this->activationFunction = activationFunction;
		randomiseWeights();
	}

	/* Takes FunctionTypes as parameter
	 */
	Node(typename FunctionTypes activationFunctiontype, int inputSize) 
	: inputSize(inputSize), weight(1, inputSize, 0) {
		this->activationFunction = Functions<T>::getFunction(activationFunctiontype);
		randomiseWeights();
	}

	/* Returns prediction for a given input
	 * input is in the for of a reference to a VMatrix of size (inputSize, 1)
	 */
	T predict(const VMatrix<T>& input) {
		T z = (weight * input).get(0,0) + bias;
		T k = activationFunction(z);
		return activationFunction(z);
	}

	/**/
};

#endif