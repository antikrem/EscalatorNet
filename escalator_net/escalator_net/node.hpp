/* Single node for a neural network
*/
#ifndef __NODE__
#define __NODE__

#include "functions.hpp"
#include "rand_ex.hpp"
#include "vmatrix.hpp"

// Forward declaraction of Node class for use by outstream operator declaraction
template <typename T>
class Node;

// Forward declaration of stream output
template <typename T> static std::ostream& operator<<(std::ostream& os, const Node<T>& n);

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

	// Finite step difference constant
	const T H = T(0.001);

	/// Learning hyper parameters
	// Learning rate 
	const T L_RATE = T(0.25);
	// Cost optimisation threshold
	const T C_THRESH = T(0.01);

	// size of input
	uint inputSize;
	
	/// Node parameters
	// Associated weights
	VMatrix<T> weight;

	// Bias for node
	T bias = 0;

	/// Forward propogation step

	// Forward propogation step seeded input input
	VMatrix<T> input = VMatrix<T>(1, 1);

	// Forward propogation step computed z
	VMatrix<T> z = VMatrix<T>(1, 1);

	// Forward propogation step computed activation
	VMatrix<T> a = VMatrix<T>(1, 1);

	/// Backwards propogation step

	// Change in weight computed after weight computation
	VMatrix<T> dWeight;

	// Change in bias computed after weight computation
	T dBias = 0;

	// Randomize weights
	void randomiseWeights() {
		rand_ex::sampleNextUniforms(weight.get(), inputSize, 0.0, 1.0);
	}

	// Declare outstream print as a friend <3
	template <typename T> friend std::ostream& operator<<(std::ostream& os, const Node<T>& n);

public:
	/* Takes FunctionTypes as parameter
	 */
	Node(typename FunctionTypes activationFunctionType, int inputSize)
	: inputSize(inputSize), weight(1, inputSize, T(0)), dWeight(1, inputSize, T(0)) {
		this->activationFunctionType = activationFunctionType;
		this->activationFunction = Functions<T>::getFunction(activationFunctionType);
		this->activationFunctionDerivative = Functions<T>::getFunctionDerivative(activationFunctionType);
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
	 * Weight is a matrix where each column is a set of weights, and there are k columns
	 * Will return a matrix of size (k, j), where each row is a prediction for the ith input
	 * and each collumn is a different 
	 */
	VMatrix<T> vPredict(const VMatrix<T>& input, const VMatrix<T>& weight, const T& bias) {
		assert(input.getRowLength() == inputSize && "Input must be accepted size of (inputSize, j)");
		VMatrix z = (input * weight) + bias;
		return z.apply(activationFunction);
	}

	/* Returns Z, which is the linear combination of input and weight + bias
	 * each row is another input
	 */
	VMatrix<T> computeZ(const VMatrix<T>& input, const VMatrix<T>& weight, const T& bias) {
		assert(input.getRowLength() == inputSize && "Input must be accepted size of (inputSize, j)");
		return (input * weight) + bias;
	}

	/* Applies activation to Z
	*/
	VMatrix<T> computeA(const VMatrix<T>& z) {
		return z.apply(activationFunction);
	}

	/* Computes rate of change for each weight
	 */
	// TODO: use central  finite difference
	VMatrix<T> computeWeightDerivatives(VMatrix<T> YObs) { 
		// A matrix that can be used to strech weight matrix into a square matrix
		VMatrix<T> stretcher = VMatrix(weight.getColumnLength(), 1, T(1.0f));

		// Creates a matrix where each column is the weight
		VMatrix<T> weightSquare = weight * stretcher;

		// Each weight collumn is a different offset for corresponding weight
		VMatrix<T> weightHSquare = weightSquare + (VMatrix<T>(weightSquare.getRowLength(), weightSquare.getColumnLength()) * H);

		// Cost at input
		VMatrix cost = vComputeCost(vPredict(input, weightSquare, bias), YObs * stretcher);
		// Cost at input + h
		VMatrix costH = vComputeCost(vPredict(input, weightHSquare, bias), YObs * stretcher);

		// Return finite difference
		return (costH - cost).qTranspose() * (T(1.0f) / H);
	}

	/* Computes rate of change for each weight
	 */
	 // TODO: use central  finite difference
	VMatrix<T> computeWeightDerivativesAnalytically(VMatrix<T> YObs) {
		// Compute each sub derivative in chain rule

		// Matrices will need to be streched to acomodate each weight
		VMatrix<T> stretcher = VMatrix(weight.getColumnLength(), 1, T(1.0f));

		// Cost relative to this nodes activation
		VMatrix<T> dCda = (a - YObs) * T(2.0) * stretcher;

		// activation relative to z
		VMatrix<T> dadz = z.apply(activationFunctionDerivative) * stretcher;

		// z relative to bias
		VMatrix<T> dzdw = input;

		return dCda.elementMultiply(dadz).elementMultiply(dzdw);
	}

	/* Computes rate of change for bias through finite difference
	*/
	T computeBiasDerivative(VMatrix<T> YObs) {
		// Creates a matrix where each column is the weight
		T biasH = bias + H;

		// Cost at input
		T cost = computeCost(vPredict(input, weight, bias), YObs);
		// Cost at input + h
		T costH = computeCost(vPredict(input, weight, biasH), YObs);

		// Return finite difference
		return (costH - cost) * (T(1.0f) / H);
	}

	/* Computes cost derivative relative to bias analytically
	 * For current values of z and a
	 */
	VMatrix<T> computeBiasDerivativeAnalytically(VMatrix<T> YObs) {
		// Compute each sub derivative in chain rule
		
		// Cost relative to this nodes activation
		VMatrix<T> dCda = (a - YObs) * T(2.0);

		// activation relative to z
		VMatrix<T> dadz = z.apply(activationFunctionDerivative);

		// z relative to bias
		T dzdb = T(1.0);

		return dadz.elementMultiply(dCda) * dzdb;
	}


	/* Computes cost for multiple inputs
	 * with existing prediction
	 */
	T computeCost(const VMatrix<T>& YPred, const VMatrix<T>& YObs) {
		// TODO: use cross-entropy cost
		return (YPred - YObs).apply(
			[](T value) {
				return pow(value, T(2));
			}
		).sum();
	}

	/* Computes multiple costs for multiple inputs
	 * with existing prediction
	 * return is of size (j, 1), with each column being a new 
	 */
	VMatrix<T> vComputeCost(const VMatrix<T>& YPred, const VMatrix<T>& YObs) {
		// TODO: use cross-entropy cost
		return (YPred - YObs).apply(
			[](T value) {
			return pow(value, 2);
		}
		).sumColumns();
	}

	/* Optimises this node's weights given an observation
	 * and internal input
	 * Returns best(last) prediction
	 */
	VMatrix<T> optimise(const VMatrix<T>& YObs) {
		assert(YObs.getColumnLength() == input.getColumnLength() && "Observation is in the form (1, j)");

		// Current prediction with given weight
		VMatrix<T> YPred = vPredict(input, weight, bias);
		T cost = computeCost(YPred, YObs);

		// Continue until cost is within threshold
		while (cost > C_THRESH) {
			
			// Utilise gradient to compute new bias
			bias = bias - computeBiasDerivative(YObs) * L_RATE;

			// Utilise gradient to compute new weights
			weight = weight - computeWeightDerivatives(YObs) * L_RATE;

			// Update cost with new weight
			YPred = vPredict(input, weight, bias);
			cost = computeCost(YPred, YObs);
		}

		return YPred;
	}

	// Conducts forwards propogation step with input X
	// Where each row is a vector of input
	// Returns activation of this node (1, j) where each row is activation for each input
	VMatrix<T> forwardPropogation(const VMatrix<T>& X) {
		// Apply gradient descent from previous back propogation
		weight = weight - dWeight;
		bias = bias - dBias;

		// Set input for backprop
		input.assign(X);

		// Compute z of this node
		z.assign(computeZ(input, weight, bias));

		// Compute activation of this node
		a.assign(computeA(z));
		return a;
	}

	// Backwards propogation step, takes required prediction
	// And optimises weight and bias by a single step
	VMatrix<T> backwardsPropogation(const VMatrix<T>& YObs) {
		// Compute change in cost relative to bias and weight
		dWeight = computeWeightDerivativesAnalytically(YObs).sumColumns().qTranspose() * T(1 / T(YObs.getColumnLength()));
		
		// Compute change in cost relative to bias and weight
		dBias = computeBiasDerivativeAnalytically(YObs).sum() * T(1 / T(YObs.getColumnLength()));
		
		return YObs;
	}

};


template <typename T>
static std::ostream& operator<<(std::ostream& os, const Node<T>& n)
{
	std::cout << "NODE: " << Functions<T>::getFunctionName(n.activationFunctionType) << std::endl;
	std::cout << "WEIGHT: " << n.weight.qTranspose() << std::endl;
	std::cout << "BIAS: " << n.bias;
	return os;
}

#endif