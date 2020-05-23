/* The network object
 */
#ifndef __NETWORK__
#define __NETWORK__

#include "layer.hpp"
#include "stopwatch.hpp"
#include "hyper_parameters.h"

 // Forward declaraction of Node class for use by outstream operator declaraction
template <typename T>
class Network;

// Forward declaration of stream output
template <typename T> static std::ostream& operator<<(std::ostream& os, const Network<T>& n);

 /* Contains a full network
  */
template <typename T>
class Network {
private:
	// Set to true upon convergence
	bool converged = false;

	// Set to true when an input comes in for an example
	bool seeded = false;

	// Internal Input
	VMatrix<T> internalInput = VMatrix <T>(1, 1);

	// Internal Output
	VMatrix<T> internalOutput = VMatrix <T>(1, 1);

	// Hyper parameters for optimisation
	HyperParameters hParams;

	// time taken
	double executionTime = 0.0;

	// number of iterations for last optimisation
	uint count = 0;

	// cost for last iteraion
	T cost = T(0);

	// List of hidden layers
	std::vector<Layer<T>> layers;

	// Last output from forward propogation
	VMatrix<T> lastPrediction = VMatrix <T>(1, 1);

	// Declare outstream print as a friend ))
	template <typename T> friend std::ostream& operator<<(std::ostream& os, const Network<T>& n);

public:
	// Generate network with given number of hidden layers
	// All using same activation function
	Network(uint inputWidth, FunctionTypes type, std::vector<uint> nodeCounts, T leaningRate = T(1.0)) {
		// Size of input for next layer
		uint nextWidth = inputWidth;
		for (auto i : nodeCounts) {
			layers.push_back(Layer<T>(nextWidth, i, type, leaningRate));
			nextWidth = i;
		}
	}

	// Generate network with given number of all layers
	// Using more default parameters
	Network(std::vector<uint> nodeCounts, std::string name)
		: Network<T>(nodeCounts[0], Functions<T>::getFunctionFromName(name), std::vector<uint>(nodeCounts.begin() + 1, nodeCounts.end())) {
	}

	// Get a reference to internal hyper parameters
	HyperParameters& getHParams() {
		return hParams;
	}

	// Forward propogates through all layers
	// Returns vector of predicted outputs from last iteration
	VMatrix<T> forwardPropogate(const VMatrix<T>& input) {

		// Input for next layer
		lastPrediction.assign(input);

		for (auto& i : layers) {
			lastPrediction.assign(i.propogateForward(lastPrediction));
		}

		return lastPrediction;
	}

	// Compute the rate of change of cost
	// relative to the activation of the output layer
	VMatrix<T> computeDCDa(const VMatrix<T> activation, const VMatrix<T> YObs) {
		return (activation - YObs) * T(2.0);
	}

	// Backwards propogation step
	// Takes Matrix where each column is the expected output
	// Of the ith node in the output layer
	// And each column corresponds to a new input
	void backwardPropogate(const VMatrix<T>& YObs) {

		// Iterate through backwards
		for (uint i = (uint)layers.size(); i--;) {
			// compute dcda
			if (i == layers.size() - 1) {
				layers[i].setFirstdcda(YObs);
			}
			else {
				layers[i].setdcda(layers[i + 1]);
			}
			layers[i].propogateBackwards();
		}
	}

	// Calculates cost given the last forward prediction
	T computeCost(const VMatrix<T>& YObs) {
		assert(lastPrediction.getRowLength() == YObs.getRowLength()
			&& lastPrediction.getColumnLength() == YObs.getColumnLength()
			&& "Observation must be the same dimensions as prediction");

		return (YObs - lastPrediction).apply(
				[](T value) {
					return pow(value, T(2));
				}
			).sum();
	}

	// Sets the training set, each row is a new example


	// Adds an example
	void addExample(const VMatrix<T>& input, const VMatrix<T>& output) {
		// if not seeded, proceed to seed
		if (!seeded) {
			internalInput.assign(input);
			internalOutput.assign(output);
			seeded = true;
		}
		else {
			internalInput.extend(input);
			internalOutput.extend(output);
		}
	}

	// Trains this network against internal input and output
	void train(bool print = false) {

		// Cache some hyper parameters
		const double CTHRESH = hParams.get(CONVERGENCE_THRESHOLD);
		const uint ITERMAX = (uint)hParams.get(ITERATION_MAX);

		// Prepare updated variables
		cost = T(CTHRESH + T(1));
		count = 0;

		stopwatch::tic();

		while (cost > CTHRESH && count < ITERMAX) {
			// capture activation from forward propogation
			VMatrix<T> activations = forwardPropogate(internalInput);
			cost = computeCost(internalOutput);

			if (print && !(count % 10000)) {
				std::cout << "Cost: " << cost << " Left: " << ITERMAX - count << std::endl;
			}

			// Apply an interation of backprop
			backwardPropogate(internalOutput);

			count++;
		}

		executionTime = stopwatch::tocGet();

		// Set convergence flag
		converged = cost < CTHRESH;
	}

	// Makes prediction with given input
	VMatrix<T> makePrediction(const VMatrix<T>& input) {
		return forwardPropogate(input);
	}

	// Makes prediction with nice output
	void predict(const VMatrix<T>& input) {
		std::cout << "Input:" << std::endl;
		std::cout << input.transpose() << std::endl;
		std::cout << "output:" << std::endl;
		std::cout << makePrediction(input).transpose() << std::endl;
	}
};

template <typename T>
static std::ostream& operator<<(std::ostream& os, const Network<T>& n)
{
	std::cout << "NETWORK of " << n.layers.size() << " layers" << std::endl;
	std::cout << "Converged: " << (n.converged ? "TRUE" : "FALSE")
		<< " IN " << n.executionTime << " seconds" << std::endl;
	std::cout << "Iterations: " << n.count << std::endl;
	std::cout << "Cost: " << n.cost;
	return os;
}

#endif