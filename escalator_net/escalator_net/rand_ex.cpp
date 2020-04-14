#include <mutex>

#include "rand_ex.hpp"

std::mutex randLock;
std::default_random_engine rGen;

// Engine to reset to
std::default_random_engine fallback;

std::default_random_engine& rand_ex::getRandomEngine() {
	return rGen;
}

void rand_ex::reset() {
	memcpy(&rGen, &fallback, sizeof(std::default_random_engine));
}