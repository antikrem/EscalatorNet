/* Small number of tests run
 * Used to check individual node performance*/
#ifndef __SINGLE_NODE__
#define __SINGLE_NODE__

#include "node.hpp"

namespace tests {
	/* Simple detector
	 * Predicts 1 on 1 input, 0 on 0 input
	 */
	void sn_detector() {
		Node a = Node<double>(FunctionTypes::ReLU, 1);

		VMatrix<double> input(1, 2, 0.0);
		input.set(0, 0, 0.0f);
		input.set(0, 1, 1.0f);


		VMatrix<double> output(1, 2, 0.0);
		output.set(0, 0, 0.0f);
		output.set(0, 1, 1.0f);
		a.forwardPropogation(input);
		auto b = a.optimise(output);

		std::cout << a << std::endl;
		std::cout << "INPUT:" << std::endl << input << std::endl;
		std::cout << "SOLUTION:" << std::endl << b << std::endl;
	}

	/* Implementation of AND gate
	 */
	void sn_ANDGate() {
		Node a = Node<double>(FunctionTypes::ReLU, 2);

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

		a.forwardPropogation(input);
		auto b = a.optimise(output);

		std::cout << a << std::endl;
		std::cout << "INPUT:" << std::endl << input << std::endl;
		std::cout << "SOLUTION:" << std::endl << b << std::endl;
	}


	/* Runs all test
	*/
	void runSingleNodeTests() {
		std::cout << "Detector:" << std::endl;
		sn_detector();
		std::cout << std::endl;
		std::cout << "AND Gate:" << std::endl;
		sn_ANDGate();
		std::cout << std::endl;

	}
}


#endif