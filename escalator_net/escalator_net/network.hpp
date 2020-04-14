/* The network object
 */
#ifndef __NETWORK__
#define __NETWORK__

#include "layer.hpp"

// Feed structure for initialising network

 /* Gradient descent optimisation results
  * No invariants, totally public
  */
template <typename T>
struct Network {
private:
	// Threshold for total network cost
	const T C_THRESH = T(0.01);

	// number of iterations for late optimisation
	uint count = 0;

	// Upperbound for total count
	const uint ITER_MAX = 200;

	// List of hidden layers
	std::vector<Layer<T>> layers;

	// Last output from forward propogation
	VMatrix<T> lastPrediction = VMatrix <T>(1, 1);

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
			lastPrediction.assign(i.propogateForward(input));
		}

		return lastPrediction;
	}

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
		T cost = T(C_THRESH + T(1));

		count = 0;

		while (cost > C_THRESH && count < ITER_MAX) {
			forwardPropogate(input);
			cost = computeCost(YObs);

			if (print) {
				std::cout << cost << std::endl;
			}

			backwardPropogate(YObs);
			count++;
		}
	}

	// Makes prediction with given input
	VMatrix<T> makePrediction(const VMatrix<T>& input) {
		return forwardPropogate(input);
	}
};

#endif