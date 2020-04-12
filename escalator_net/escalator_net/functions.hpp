/* static class that encapsulates functions for a 
*/
#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "types.hpp"
#include <algorithm>

// 
enum FunctionTypes {
	sigmoid,
	ReLU,
	linear
};

// List of templated functions for use
template <typename T>
class Functions {
public:
	// function ptr for this type of function
	using function_ptr = T(*)(T);

	// Implementation of sigmoid function
	static T sigmoid(T x) {
		return 1 / (1 + exp(-x));
	}

	// Implementation of sigmoid derivative
	static T sigmoidDerivative(T x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}

	// Implementation of ReLU
	static T ReLU(T x) {
		return x < T(0) ? 0 : x;
	}

	// Implementation of linear pass
	static T linear(T x) {
		return x < T(0) ? 0 : x;
	}

	// Returns corresponding function
	static function_ptr getFunction(FunctionTypes type) {
		switch (type) {
		case FunctionTypes::sigmoid:
			return Functions::sigmoid;

		case FunctionTypes::ReLU:
			return Functions::ReLU;

		case FunctionTypes::linear:
			return Functions::linear;

		default:
			return nullptr;
		}
	}

	// Returns corresponding function derivative
	static function_ptr getFunctionDerivative(FunctionTypes type) {
		switch (type) {
		case FunctionTypes::sigmoid:
			return Functions::sigmoidDerivative;

		default:
			return nullptr;
		}
	}
};

#endif