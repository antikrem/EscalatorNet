/* Header for templated algorithms
*/
#ifndef __ALG__
#define __ALG__

#include "types.hpp"
#include <algorithm>

namespace alg {
	// Fill for c-style array of fixed size
	template <typename T>
	void fill(T arr[], uint length, T value) {
		std::fill(arr, arr + length, value);
	}

	// Implementation of copy for array with fixed size
	template <typename T>
	void copy(T source[], T destination[], uint length) {
		std::copy(source, source + length, destination);
	}
}

#endif