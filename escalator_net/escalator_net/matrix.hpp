/* Templated implementation of matrix
 * is statically generated
 * Has fixed size
 */

#ifndef __MATRIX__
#define __MATRIX__

#include <assert.h>
#include <functional>

#include "types.hpp"
#include "alg.hpp"

/* Matrix contains 3 templated properies: 
 * T: The type of each cell
 * R, C: The number of rows and number of columns respectivly 
 * Has support for basic matrix operations
 */
template <uint R, uint C, typename T = double>
class Matrix {
	
public:
	// Templated parameters
	static constexpr uint ROW_LENGTH = R;
	static constexpr uint COLUMN_LENGTH = C;
	static constexpr uint LENGTH = C * R;

private:
	/* Data is stored on the stack if Matrix is initialised as a stack variable
	 * 
	 * Translation to get index i from (x, y), where x is how far to the right, and y is how far down:
	 * F: (x,y) -> i : y * C + x
	 * 
	 * Inverse translation, index to (x,y):
	 * F^-1: i -> (x,y) : (i / C, i % c)
	 */
	T data[LENGTH];

	// Constructor for matrix via identity
	// Only works for square matrix
public:
	// Set value by (x,y) coord
	// Throws assertion error on invalid x,y in debug
	void set(uint x, uint y, T value) {
		assert(x < ROW_LENGTH && y < COLUMN_LENGTH && "Attempt to index outside of matrix range");
		data[y * ROW_LENGTH + x] = value;
	}

	// Set value by (x,y) coord
	// is templated
	template<uint X, uint Y>
	void set(T value) {
		static_assert(X < ROW_LENGTH && Y < COLUMN_LENGTH, "Attempt to index outside of matrix range");
		data[Y * ROW_LENGTH + X] = value;
	}

	// get value by (x,y) coord
	// Throws assertion error on invalid x,y in debug
	T get(uint x, uint y) const {
		assert(x < ROW_LENGTH && y < COLUMN_LENGTH && "Attempt to index outside of matrix range");
		return data[y * ROW_LENGTH + x];
	}

	// get value by (x,y) coord
	// is templated
	template<uint X, uint Y>
	T get() const {
		static_assert(X < ROW_LENGTH && Y < COLUMN_LENGTH, "Attempt to index outside of matrix range");
		return data[Y * ROW_LENGTH + X];
	}

	// Generates identity matrix
	constexpr Matrix() {
		static_assert(ROW_LENGTH == COLUMN_LENGTH, "Identity matrix only supported for square matrix");
		alg::fill(data, LENGTH, T(0));
		for (int i = 0; i < ROW_LENGTH; i++) {
			set(i, i, T(1));
		}
	}

	// Generates matrix and fills values
	constexpr Matrix(T value) {
		alg::fill(data, LENGTH, value);
	}

	// Generates values for matrix with a lambda f, mapping row, width to a value
	// f : (x:int, y:int). -> v:T
	Matrix(std::function<T(uint, uint)> generator) {
		for (uint i = 0; i < ROW_LENGTH; i++) {
			for (uint j = 0; j < COLUMN_LENGTH; j++) {
				set(i, j, generator(i, j));
			}
		}
	}

};

#endif