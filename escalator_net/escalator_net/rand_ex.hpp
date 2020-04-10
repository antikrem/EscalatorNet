/* Header for templated algorithms
*/
#ifndef __RAND_EX__
#define __RAND_EX__

#include <algorithm>
#include <random>

#include "types.hpp"

namespace rand_ex {
	// Gets reference to random engine
	std::default_random_engine& getRandomEngine();

	// Returns next instance of a given template
	template<typename T>
	T sampleNextUniform(T a, T b) {
		std::uniform_real_distribution<T> uDist(a, b);
		return uDist(getRandomEngine());
	}

	// Sets variable of given length to fill with 
	template<typename T>
	void sampleNextUniforms(T* array, uint length, T a, T b) {
		std::uniform_real_distribution<T> uDist(a, b);
		for (uint i = 0; i < length; i++) {
			array[i] = uDist(getRandomEngine());
		}
	}
}

#endif