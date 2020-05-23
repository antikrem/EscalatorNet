/* static class that encapsulates functions for a 
*/
#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include <algorithm>
#include <string>
#include <map>
#include <math.h> 

#include "types.hpp"


// Different types of activation functions 
enum FunctionTypes {
	none,
	sigmoid,
	ReLU,
	LeakyReLU,
	softplus,
};

// Map of FunctionTypes to string name
const std::map<FunctionTypes, std::string> FUNCTION_TYPES_TO_NAME = {
	{sigmoid, "sigmoid"},
	{ReLU, "ReLU"},
	{LeakyReLU, "LeakyReLU"},
	{softplus, "softplus"}
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

	// Implementation of LeakyReLU
	static T LeakyReLU(T x) {
		return x < T(0) ? 0.1 * x : x;
	}

	// Implementation of LeakyReLU derivative
	static T LeakyReLUDerivative(T x) {
		return x < T(0) ? T(0.1) : T(1);
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

		case FunctionTypes::LeakyReLU:
			return Functions::LeakyReLU;

		case FunctionTypes::softplus:
			return Functions::softplus;

		default:
			return nullptr;
		}
	}

	// Returns a string name to a corresponding function type
	static std::string getFunctionName(FunctionTypes type) {
		if (FUNCTION_TYPES_TO_NAME.count(type)) {
			return FUNCTION_TYPES_TO_NAME[type];
		}
		else {
			return "None";
		}
	}

	// Returns function type from a name
	static FunctionTypes getFunctionFromName(std::string name) {
		for (auto& i : FUNCTION_TYPES_TO_NAME) {
			if (i.second == name) {
				return i.first;
			}
		}
		return FunctionTypes::none;
	}

	// Returns corresponding function derivative
	static function_ptr getFunctionDerivative(FunctionTypes type) {
		switch (type) {
		case FunctionTypes::sigmoid:
			return Functions::sigmoidDerivative;
		
		case FunctionTypes::ReLU:
			return Functions::ReLUDerivative;

		case FunctionTypes::LeakyReLU:
			return Functions::LeakyReLUDerivative;

		case FunctionTypes::softplus:
			return Functions::softplusDerivative;

		default:
			return nullptr;
		}
	}
};

#endif