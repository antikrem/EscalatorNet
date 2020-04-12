// escalator_net.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// If defined, compilation will run tests
#define RUN_TESTS

#include <iostream>

#ifdef RUN_TESTS 
#include "_tests.h"
#endif // RUN_TESTS

int main()
{

#ifdef RUN_TESTS 
	tests::runAllTests();

	system("pause");
#endif // RUN_TESTS
	
}
