/* Can be added during compilation to run a number of tests*/
#ifndef ___TESTS__
#define ___TESTS__

#include <iostream>
#include "single_node.hpp"

namespace tests {
	/* Prints small message about which tests should be run
	 */
	void declareTest(std::string name) {
		std::cout << "-- RUNNING TESTS: " << name << " --" << std::endl;
	}

	/* Runs all tests
	 */
	void runAllTests() {
		declareTest("SINGLE_NODES");
		runSingleNodeTests();
	}
}


#endif