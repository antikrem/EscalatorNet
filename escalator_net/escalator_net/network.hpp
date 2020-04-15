/* The network object
 */
#ifndef __NETWORK__
#define __NETWORK__

#include "layer.hpp"
#include "stopwatch.hpp"

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
	// Threshold for total network cost
	const T C_THRESH = T(0.01);

	// Upperbound for total count
	const uint ITER_MAX = 2000;

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
	Network(uint inputWidth, FunctionTypes type, std::vector<uint> nodeCounts) {
		// Size of input for next layer
		uint nextWidth = inputWidth;
		for (auto i : nodeCounts) {
			layers.push_back(Layer<T>(nextWidth, i, type));
			nextWidth = i;
		}
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

	// Backwards propogation step
	// Takes Matrix where each column is the expected output
	// Of the ith node in the output layer
	// And each column corresponds to a new input
	VMatrix<T> backwardPropogate(const VMatrix<T>& YObs) {

		for (uint i = layers.size(); i--;) {
			layers[i].propogateBackwards(YObs);
		}

		return YObs;
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

	// Optimises this network given input and observed
	void optimiseNetwork(const VMatrix<T>& input, const VMatrix<T>& YObs, bool print = false) {
		cost = T(C_THRESH + T(1));

		count = 0;

		stopwatch::tic();

		while (cost > C_THRESH && count < ITER_MAX) {
			forwardPropogate(input);
			cost = computeCost(YObs);

			if (print) {
				std::cout << cost << std::endl;
			}

			backwardPropogate(YObs);
			count++;
		}

		executionTime = stopwatch::tocGet();
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
	std::cout << "Converged: " << (n.cost < n.C_THRESH ? "TRUE" : "FALSE") 
		<< " IN " << n.executionTime << " seconds" << std::endl;
	std::cout << "Iterations: " << n.count << std::endl;
	std::cout << "Cost: " << n.cost;
	return os;
}

#endif