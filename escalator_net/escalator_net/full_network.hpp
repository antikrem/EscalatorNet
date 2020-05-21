/* Small number of tests run
 * Used to check networks of full network*/
#ifndef __FULL_NETWORK__
#define __FULL_NETWORK__

#include "network.hpp"

namespace tests {
	/* implementation of an xor gate
	 */
	void fn_XORGate() {
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

		net.addExample(input, output);
		net.train();

		std::cout << net << std::endl;
		net.predict(input);
	}

	/* implementation of an xor gate
	 */
	void fn_NXORGate() {
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
				{1.0},
				{0.0},
				{0.0},
				{1.0}
			}
		);

		net.addExample(input, output);
		net.train();

		std::cout << net << std::endl;
		net.predict(input);
	}

	/* implementation of an xor gate, but deep
	 */
	void fn_XORGateDeep() {
		// 14 sigmoid  
		Network<double> net(2, FunctionTypes::sigmoid, { 2, 2, 1 }, 1.0);

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

		net.addExample(input, output);
		net.train();

		std::cout << net << std::endl;
		net.predict(input);
	}

	/* Runs all test
	*/
	void runFullNetworkTests() {
		std::cout << "XOR Gate:" << std::endl;
		fn_XORGate();
		std::cout << std::endl;

		std::cout << "NXOR Gate:" << std::endl;
		fn_NXORGate();
		std::cout << std::endl;

		std::cout << "XOR Deep Gate:" << std::endl;
		fn_XORGateDeep();
		std::cout << std::endl;

	}
}


#endif