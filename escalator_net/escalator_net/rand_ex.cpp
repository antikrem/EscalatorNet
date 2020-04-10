#include <mutex>

#include "rand_ex.hpp"

std::mutex randLock;
std::default_random_engine rGen;


std::default_random_engine& rand_ex::getRandomEngine() {
	return rGen;
}