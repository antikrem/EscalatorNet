/* Structure that keeps a map of hyper parameters
*/
#ifndef __HYPER_PARAMETERS__
#define __HYPER_PARAMETERS__

#include <string>
#include <map>

#define CONVERGENCE_THRESHOLD "convergence_threshold"
#define ITERATION_MAX "iteration_max"
#define LEARNING_RATE "learning_rate"

class HyperParameters {
	// Internal parameters
	std::map<std::string, double> parameters = {
		{ CONVERGENCE_THRESHOLD, 0.01 },
		{ ITERATION_MAX, 3000000.0 },
		{ LEARNING_RATE, 1.0 },
	};
	// TODO make strict
public:
	// Sets a parameter
	void set(std::string name, double value) {
		parameters[name] = value;
	}

	// Gets a parameter
	double get(std::string name) const {
		return (*parameters.find(name)).second;
	}
};

#endif