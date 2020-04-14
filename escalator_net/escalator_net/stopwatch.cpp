#include "stopwatch.hpp"

#define MICROSECOND_IN_SECOND 1000000.0

stopwatch::timepoint start = stopwatch::clock::now();

void stopwatch::tic() {
	start = clock::now();
}

stopwatch::duration stopwatch::tocRaw() {
	return (clock::now() - start);
}

double stopwatch::tocGet() {
	return 
		((double)std::chrono::duration_cast<std::chrono::microseconds>(stopwatch::tocRaw()).count() 
			/ MICROSECOND_IN_SECOND);
}