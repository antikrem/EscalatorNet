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

	// Activation for this layer from last forward propogation
	VMatrix<T> activation = VMatrix<T>(1, 1, T(0.0));

public:
	/* Computes dCda for this layer,
	 * each row is the next input, each column, dcda for ith node
	 * Takes:
	 * reference to previous layer (L+1) for values
	 */
	void setdcda(const Layer<T>& layer) {
		
		// go through each node, and compute dcda
		int i = -1;
		for (auto& node : nodes) {
			i++;
			VMatrix<T> dcda(1, node.getActivation().getColumnLength(), T(0.0));
		
			for (auto& nextNode : layer.nodes) {
				dcda = dcda + nextNode.getdcda().elementMultiply(nextNode.getdadz()) * nextNode.getWeight(i);
			}

			node.setdcda(dcda);
		}
	}

	/* Sets dcda for the first layer
	 * Takes as input, YObs, for this layer
	 * sets dcda against 
	 */
	void setFirstdcda( const VMatrix<T> YObs) {
		VMatrix<T> dcda = (activation - YObs) * T(2.0);
		for (uint i = 0; i < nodes.size(); i++) {
			nodes[i].setdcda(dcda.getColumn(i));
		}
	}

	// Creates this layer for a fixed sized input
	// with given count of of nodes and activation function
	Layer(uint inputSize, uint nodeCount, FunctionTypes activationFunction, T learningRate = T(1.0))
	: INPUTSIZE(inputSize) {
		for (uint i = 0; i < nodeCount; i++) {
			nodes.push_back(Node<T>(activationFunction, inputSize, learningRate));
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
		// Results matrix, will be extended transposed
		VMatrix<T> results(input.getColumnLength(), 0, T(0.0));

		for (Node<T>& node : nodes) {
			results.extend(node.forwardPropogation(input).qTranspose());
		}

		activation.assign(results.transpose());

		return activation;
	}

	/* Applies backward propogation step
	 * Takes dcda to current layer
	 * where each row is a new input
	 * and each column is the corresponding node's activation rate of change
	 */
	void propogateBackwards() {

		// Apply backprop to each node
		// All parameters have already been set
		for (uint i = 0; i < nodes.size(); i++) {
			nodes[i].backwardsPropogation();
		}

	}

	// Returns activation from last forward pass
	// For ith node in jth input
	VMatrix<T> getActivation() {
		return activation;
	}
};

#endif