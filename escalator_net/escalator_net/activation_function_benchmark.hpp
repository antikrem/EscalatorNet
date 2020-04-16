/* Small number of tests run
 * Used to check networks of full network*/
#ifndef __ACTIVATION_FUNCTION_BENCHMARK__
#define __ACTIVATION_FUNCTION_BENCHMARK__

#include "network.hpp"

namespace tests {

	/* implementation of an xor gate sigmoid
	 */
	void afb_XORsigmoid() {
		Network<double> net(2, FunctionTypes::sigmoid, { 2, 1 }, 1.0);

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{0.0, 1.0},
				{1.0, 0.0},
				{1.0, 1.0}
			}
		);

		VMatrix<double> output(
			{
				{0.0},
				{1.0},
				{1.0},
				{0.0}
			}
		);

		net.optimiseNetwork(input, output);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* implementation of an xor gate softplus
	 */
	void afb_XORsoftplus() { 
		Network<double> net(2, FunctionTypes::softplus, { 2, 1 }, 1.0);

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{0.0, 1.0},
				{1.0, 0.0},
				{1.0, 1.0}
			}
		);

		VMatrix<double> output(
			{
				{0.0},
				{1.0},
				{1.0},
				{0.0}
			}
		);

		net.optimiseNetwork(input, output);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* implementation of an xor gate
	 */
	void afb_XORReLU() { 
		Network<double> net(2, FunctionTypes::ReLU, { 2, 1 }, 1.0);

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{0.0, 1.0},
				{1.0, 0.0},
				{1.0, 1.0}
			}
		);

		VMatrix<double> output(
			{
				{0.0},
				{1.0},
				{1.0},
				{0.0}
			}
		);

		net.optimiseNetwork(input, output);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* implementation of an xor gate
	 */
	void afb_XORReLU() {
		Network<double> net(2, FunctionTypes::ReLU, { 2, 1 }, 1.0);

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{0.0, 1.0},
				{1.0, 0.0},
				{1.0, 1.0}
			}
		);

		VMatrix<double> output(
			{
				{0.0},
				{1.0},
				{1.0},
				{0.0}
			}
		);

		net.optimiseNetwork(input, output);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* Runs all test
	*/
	void runActivationFunctionBenchmark() {
		std::cout << "XORsigmoid Gate:" << std::endl;
		afb_XORsigmoid();
		std::cout << std::endl;

		std::cout << "XORsoftplus Gate:" << std::endl;
		afb_XORsoftplus();
		std::cout << std::endl;

		std::cout << "XORReLU Gate:" << std::endl;
		afb_XORReLU();
		std::cout << std::endl;

	}
}


#endif