/* static class that encapsulates functions for a 
*/
#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include <algorithm>
#include <string>
#include <math.h> 

#include "types.hpp"


// Different types of activation functions 
enum FunctionTypes {
	sigmoid,
	ReLU,
	softplus,
};

// List of templated functions for use
template <typename T>
class Functions {
public:
	// function ptr for this type of function
	using function_ptr = T(*)(T);

	// Implementation of sigmoid function
	static T sigmoid(T x) {
		return T(1) / (T(1) + exp(-x));
	}

	// Implementation of sigmoid derivative
	static T sigmoidDerivative(T x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}

	// Implementation of ReLU
	static T ReLU(T x) {
		return x < T(0) ? T(0) : x;
	}

	// Implementation of ReLU derivative
	static T ReLUDerivative(T x) {
		return x < T(0) ? T(0) : T(1);
	}

	// Implementation of softplus
	static T softplus(T x) {
		return log(T(1.0) + exp(x));
	}

	// Implementation of softplus derivative
	static T softplusDerivative(T x) {
		return sigmoid(x);
	}

	// Returns corresponding function
	static function_ptr getFunction(FunctionTypes type) {
		switch (type) {
		case FunctionTypes::sigmoid:
			return Functions::sigmoid;

		case FunctionTypes::ReLU:
			return Functions::ReLU;

		case FunctionTypes::softplus:
			return Functions::softplus;

		default:
			return nullptr;
		}
	}

	// Returns a string name to a corresponding function type
	static std::string getFunctionName(FunctionTypes type) {
		switch (type) {
		case FunctionTypes::sigmoid:
			return "sigmoid";

		case FunctionTypes::ReLU:
			return "ReLU";

		case FunctionTypes::softplus:
			return "softplus";

		default:
			return "None";
		}
	}

	// Returns corresponding function derivative
	static function_ptr getFunctionDerivative(FunctionTypes type) {
		switch (type) {
		case FunctionTypes::sigmoid:
			return Functions::sigmoidDerivative;
		
		case FunctionTypes::ReLU:
			return Functions::ReLUDerivative;

		case FunctionTypes::softplus:
			return Functions::softplusDerivative;

		default:
			return nullptr;
		}
	}
};

#endif