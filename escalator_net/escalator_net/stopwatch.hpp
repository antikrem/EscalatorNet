/* Header for static timer
 * similar to Matlab style tic-toc usage 
 * No garuntee on thread safety
 */
#ifndef __STOPWATCH__
#define __STOPWATCH__

#include <chrono>

#include "types.hpp"

namespace stopwatch {
	// Type defs for readability
	using clock = std::chrono::high_resolution_clock;
	using timepoint = std::chrono::time_point<clock>;
	using duration = std::common_type_t<clock::duration, clock::duration>;

	// Sets start of timer
	void tic();

	// Stops timer, returns a chrono duration of time since last tic
	// or application start
	// Returns a high resolution duration
	duration tocRaw();

	// Similar to toc raw, but casts into a second count
	double tocGet();

}

#endif