/* Small number of tests run
 * Used to check networks of single node, single layer*/
#ifndef __SHALLOW_NETWORK__
#define __SHALLOW_NETWORK__

#include "network.hpp"

namespace tests {
	/* Simple detector
	 * Predicts 1 on 1 input, 0 on 0 input
	 */
	void shn_detector() {
		Network<double> net(1, FunctionTypes::LeakyReLU, { 1 });

		VMatrix<double> input(1, 2, 0.0);
		input.set(0, 0, 0.0f);
		input.set(0, 1, 1.0f);


		VMatrix<double> output(1, 2, 0.0);
		output.set(0, 0, 0.0f);
		output.set(0, 1, 1.0f);
		
		net.optimiseNetwork(input, output);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* Implementation of AND gate
	 */
	void shn_ANDGate() {
		Network<double> net(2, FunctionTypes::LeakyReLU, { 1 });

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{1.0, 0.0},
				{0.0, 1.0},
				{1.0, 1.0}
			}
		);

		VMatrix<double> output(
			{
				{0.0},
				{0.0},
				{0.0},
				{1.0}
			}
		);

		net.optimiseNetwork(input, output);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* Implementation of OR gate
	 */
	void shn_ORGate() {
		Network<double> net(2, FunctionTypes::LeakyReLU, { 1 });

		VMatrix<double> input(
			{
				{0.0, 0.0},
				{1.0, 0.0},
				{0.0, 1.0},
				{1.0, 1.0}
			}
		);

		VMatrix<double> output(
			{
				{0.0},
				{1.0},
				{1.0},
				{1.0}
			}
		);

		net.optimiseNetwork(input, output);
		std::cout << net << std::endl;
		net.predict(input);
	}

	/* Runs all test
	*/
	void runShallowNodeTests() {
		std::cout << "Detector:" << std::endl;
		shn_detector();
		std::cout << std::endl;

		std::cout << "AND Gate:" << std::endl;
		shn_ANDGate();
		std::cout << std::endl;

		std::cout << "OR Gate:" << std::endl;
		shn_ORGate();
		std::cout << std::endl;
	}
}


#endif