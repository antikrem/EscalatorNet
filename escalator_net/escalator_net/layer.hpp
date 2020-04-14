/* Represents a layer in the network
 */
#ifndef __LAYER__
#define __LAYER__

#include <vector>

#include "node.hpp"

 /* A single layer of nodes
  */
template <typename T>
class Layer {
private:
	// input size of this node
	const uint INPUTSIZE;

	// Nodes within this layer
	std::vector<Node<T>> nodes;

public:
	// Creates this layer for a fixed sized input
	// with given count of of nodes and activation function
	Layer(uint inputSize, uint nodeCount, FunctionTypes activationFunction)
	: INPUTSIZE(inputSize) {
		for (uint i = 0; i < nodeCount; i++) {
			nodes.push_back(Node<T>(activationFunction, inputSize));
		}
	}
	
	/* Applies forward propogation step
	 * Takes array of input 
	 * each column is the i-th term in an input
	 * and each row is a new input
	 * returns matrix where each row is an input
	 * and each column is the return from i-th node
	 */
	VMatrix<T> propogateForward(const VMatrix<T>& input) {
		assert(INPUTSIZE == input.getRowLength());
		// Results matrix
		VMatrix<T> results(input.getColumnLength(), 0, T(0.0));

		for (Node<T>& node : nodes) {
			results.extend(node.forwardPropogation(input).qTranspose());
		}

		return results.transpose();
	}

	/* Applies backward propogation step
	 * Takes array of expected output
	 * Where each row is a new input
	 * And each column is the expected output from the ith node in this layer
	 */
	void propogateBackwards(const VMatrix<T>& YObs) {
		// Check there is a row for each 
		assert(nodes.size() == YObs.getRowLength());

		for (uint i = 0; i < nodes.size(); i++) {
			nodes[i].backwardsPropogation(YObs.getColumn(i));
		}
	}
};

#endif