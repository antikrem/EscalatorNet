/* Small number of tests run
 * Used to check individual layer*/
#ifndef __SINGLE_LAYER__
#define __SINGLE_LAYER__

#include "network.hpp"

namespace tests {
	/* Simple detector
	 * Predicts 1 on 1 input, 0 on 0 input
	 */
	void sl_layeroutput () {
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


		Layer<double> a(2, 2, FunctionTypes::ReLU);
		std::cout << a.propogateForward(input, 1.0) << std::endl;
	}

	/* Runs all test
	*/
	void runSingleLayerTests() {
		std::cout << "Single layer output:" << std::endl;
		sl_layeroutput();
		std::cout << std::endl;
	}
}


#endif