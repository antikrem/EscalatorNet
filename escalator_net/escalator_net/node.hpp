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

	// Finite step difference constant
	const T H = T(0.001);

	/// Learning hyper parameters
	// Learning rate 
	const T L_RATE;
	// Cost optimisation threshold
	const T C_THRESH = T(0.01);

	// size of input
	uint inputSize;
	
	/// Node parameters
	// Associated weights
	// Presented as a vertical vector
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

	// Change in each activation relative to linear combination
	VMatrix<T> dadz = VMatrix<T>(1, 1);

	// Change in cost relative to weight, for each input
	VMatrix<T> dWeightV = VMatrix<T>(1, 1, T(0));

	// Change in cost relative to weight, averaged for each 
	// Horrizontal vector
	VMatrix<T> dWeight;

	// Change in cost relative to bias, for each input
	VMatrix<T> dBiasV = VMatrix<T>(1, 1, T(0));

	// Change in cost relative to bias, averaged
	T dBias = 0;

	// Change in cost relative to each component in input
	// for each input
	VMatrix<T> dcda;

	// Randomize weights
	void randomiseWeights() {
		rand_ex::sampleNextUniforms(weight.qGet(), inputSize, 0.0, 1.0);
	}

	// Declare outstream print as a friend <3
	template <typename T> friend std::ostream& operator<<(std::ostream& os, const Node<T>& n);

public:
	/* Takes FunctionTypes as parameter
	 */
	Node(typename FunctionTypes activationFunctionType, int inputSize, T leaningRate = T(1.0))
	: inputSize(inputSize), weight(1, inputSize, T(0)), dWeight(inputSize, 1, T(0)), dcda(1, 1, T(0)), L_RATE(leaningRate) {
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
	 * Input is a matrix where each row is a set of input and each column is the i-th element in input
	 * weight is a single width vertical vector, and bias a scalar
	 */
	VMatrix<T> computeZ(const VMatrix<T>& input, const VMatrix<T>& weight, const T& bias) {
		assert(input.getRowLength() == inputSize && "Input must be accepted size of (inputSize, j)");
		return (input * weight) + bias;
	}

	/* Applies activation function to z
	 * Takes a matrix of given size, and return same size matrix
	 */
	VMatrix<T> computeA(const VMatrix<T>& z) {
		return z.apply(activationFunction);
	}

	/* Computes rate of change for each weight
	 */
	VMatrix<T> computeDCostDWeight(const VMatrix<T>& dCdaIN, const VMatrix<T>& dadzIN) {
		// Compute each sub derivative in chain rule
		// Recall:
		// dC/dw = dz/dw da/dz dC/da

		// Matrices will need to be streched to acomodate each weight
		VMatrix<T> stretcher = VMatrix(weight.getColumnLength(), 1, T(1.0f));

		// Cost relative to this nodes activation
		VMatrix<T> dCda = dCdaIN * stretcher;

		// activation relative to z
		VMatrix<T> dadz = dadzIN * stretcher;

		// z relative to bias
		VMatrix<T> dzdw = input;

		return dCda.elementMultiply(dadz).elementMultiply(dzdw);
	}

	/* Computes cost derivative relative to bias analytically
	 * For current values of z and a
	 */
	VMatrix<T> computeDCostDBias(const VMatrix<T>& dCdaIN, const VMatrix<T>& dadzIN) {
		// Compute each sub derivative in chain rule
		// Recall:
		// dC/db = dz/db da/dz dC/da
		
		// Cost relative to this nodes activation
		const VMatrix<T>& dCda = dCdaIN;

		// activation relative to z
		const VMatrix<T>& dadz = dadzIN;

		// z relative to bias
		T dzdb = T(1.0);

		return dadz.elementMultiply(dCda) * dzdb;
	}

	/* Computes cost derivative for this node, for each input
	 */
	VMatrix<T> computeDCostDinput(const VMatrix<T>& dCdaIN, const VMatrix<T>& dadzIN) {
		// Compute each sub derivative in chain rule
		// Recall:
		// dC/dIn = dz/dIn da/dz dC/da

		// Matrices will need to be streched to acomodate each weight
		VMatrix<T> stretcher = VMatrix(weight.getColumnLength(), 1, T(1.0f));

		// Cost relative to this nodes activation
		VMatrix<T> dCda = dCdaIN * stretcher;

		// activation relative to z
		VMatrix<T> dadz = dadzIN * stretcher;

		// z relative to input
		VMatrix<T> dzdIn = input;

		return dCda.elementMultiply(dadz).elementMultiply(dzdIn).sumRows();
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

	/* Conducts forwards propogation step with input X
	 * Where each row is a vector of input
	 * Returns activation of this node (1, j) where each row is activation for each input
	 */
	VMatrix<T> forwardPropogation(const VMatrix<T>& X) {
		// Apply gradient descent from previous back propogation
		weight = weight - dWeight.qTranspose();
		bias = bias - dBias;

		// Set input for backprop
		input.assign(X);

		// Compute z of this node
		z.assign(computeZ(input, weight, bias));

		// compute shared sub-derivative dadz
		dadz.assign(z.apply(activationFunctionDerivative));


		// Compute activation of this node
		a.assign(computeA(z));
		return a;
	}

	/* Returns reference to activation
	 */
	const VMatrix<T>& getActivation() {
		return a;
	}

	/* Backwards propogation step, takes required prediction
	 * And optimises weight and bias by a single step
	 */
	void backwardsPropogation() {
		// Compute gradients of cost for parameters for gradient descent
		// Gradient will be averaged over each input set

		// Compute change in cost relative to bias and weight
		dWeightV.assign(computeDCostDWeight(dcda, dadz));
		dWeight = dWeightV.sumColumns() * T(L_RATE / T(dcda.getColumnLength()));

		// Compute change in cost relative to bias and weight
		dBiasV.assign(computeDCostDBias(dcda, dadz));
		dBias = dBiasV.sum() * T(L_RATE / T(dcda.getColumnLength()));
	}

	/* sets dcda: rate of change of cost given this nodes activation
	 * in the form of a column vector
	 */
	void setdcda(const VMatrix<T> dcda) {
		this->dcda.assign(dcda);
	}

	/* sets dcda: rate of change of cost given this nodes activation
	 * in the form of a column vector
	 */
	const VMatrix<T>& getdcda() const {
		return dcda;
	}

	/* Gets dadz from last forward propogation
	 */
	const VMatrix<T>& getdadz() const {
		return dadz;
	}

	/* Gets input, where each column is the for the kth prior node
	 * and each row is a new input
	 */
	T getWeight(uint i) const {
		return weight.qGet(i);
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