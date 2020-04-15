/* Small number of tests run
 * Used to check networks of full network*/
#ifndef __FULL_NETWORK__
#define __FULL_NETWORK__

#include "network.hpp"

namespace tests {
	/* Simple feed through test
	 */
	void fn_feedthrough22() {
		Network<double> net(2, FunctionTypes::softplus, { 2, 2 });

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{1.0, 0.0},
				{0.0, 1.0},
				{1.0, 1.0}
			}
		);

		std::cout << net.forwardPropogate(input) << std::endl;
	}

	/* Simple feed through test
	 */
	void fn_feedthrough321() {
		Network<double> net(3, FunctionTypes::ReLU, { 3, 2, 1 });

		VMatrix<double> input(
			{
				{0.0, 0.0, 1.0},
				{1.0, 0.0, 1.0},
				{0.0, 1.0, 0.0},
				{1.0, 1.0, 0.0}
			}
		);

		std::cout << net.forwardPropogate(input) << std::endl;
	}

	/* Simple feed through test
	 */
	void fn_feedthrough353() {
		Network<double> net(2, FunctionTypes::sigmoid, { 3, 5, 3 });

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{0.25, 0.25},
				{0.75, 1.0},
				{1.45, 1.54}
			}
		);

		std::cout << net.forwardPropogate(input) << std::endl;
	}

	/* implementation of an xor gate
	 */
	void fn_XORGate() {
		// 14 sigmoid
		Network<double> net(2, FunctionTypes::sigmoid, { 2, 1 }, 10.0);

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

		net.optimiseNetwork(input, output, true);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* implementation of an xor gate
	 */
	void fn_NXORGate() {
		// 14 sigmoid
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
				{1.0},
				{0.0},
				{0.0},
				{1.0}
			}
		);

		net.optimiseNetwork(input, output, true);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* Runs all test
	*/
	void runFeedForwardTests() {
		std::cout << "Feed forward 2-2:" << std::endl;
		fn_feedthrough22();
		std::cout << std::endl;

		std::cout << "Feed forward 3-2-1:" << std::endl;
		fn_feedthrough321();
		std::cout << std::endl;

		std::cout << "Feed forward 3-5-3:" << std::endl;
		fn_feedthrough353();
		std::cout << std::endl;

		std::cout << "XOR Gate:" << std::endl;
		fn_XORGate();
		std::cout << std::endl;

		std::cout << "NXOR Gate:" << std::endl;
		fn_NXORGate();
		std::cout << std::endl;

	}
}


#endif