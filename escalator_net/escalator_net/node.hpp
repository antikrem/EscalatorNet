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
	// Activation function type
	FunctionTypes activationFunctionType;

	typename Functions<T>::function_ptr activationFunction = nullptr;
	typename Functions<T>::function_ptr activationFunctionDerivative = nullptr;

	// Threshold for a node activation
	const T activationThreshold = T(0.5);

	// size of input
	uint inputSize;
	
	/// Node parameters
	// Associated weights
	VMatrix<T> weight;

	// Bias for node
	T bias = 0;

	///calculated forward propogation results
	// Randomize weights
	void randomiseWeights() {
		rand_ex::sampleNextUniforms(weight.get(), inputSize, 0.0, 1.0);
	}

public:
	/* Takes FunctionTypes as parameter
	 */
	Node(typename FunctionTypes activationFunctiontype, int inputSize) 
	: inputSize(inputSize), weight(1, inputSize, 0) {
		this->activationFunctionType = activationFunctionType;
		this->activationFunction = Functions<T>::getFunction(activationFunctiontype);
		randomiseWeights();
	}

	/* Returns prediction for a given input
	 * input is in the for of a reference to a VMatrix of size (inputSize, 1)
	 */
	T predict(const VMatrix<T>& input) {
		assert(input.getRowLength() == inputSize && input.getColumnLength() == 1 && "Input must be accepted size of (inputSize, 1)");
		T z = (input * weight).get(0,0) + bias;
		T k = activationFunction(z);
		return activationFunction(z);
	}

	/* Returns prediction for a given input
	 * input is in the for of a reference to a VMatrix of size (inputSize, j)
	 * Will return a matrix of size (1, j)
	 */
	VMatrix<T> vPredict(const VMatrix<T>& input) {
		assert(input.getRowLength() == inputSize && "Input must be accepted size of (inputSize, j)");
		VMatrix z = (input * weight) + bias;
		auto k = z.apply(activationFunction);
		return z.apply(activationFunction);
	}


	/* Computes the cost of an input, in the form
	 * X(i,j) : where each row is a new observation of inputs
	 * Y(1,j) : where each row is a expected output
	 */
	void optimise(const VMatrix<T>& X, const VMatrix<T>& Y) {

	}
};

#endif