/* Runtime implementation of matrix
 * is statically generated
 * Has runtime specified size
 */

#ifndef __VMATRIX__
#define __VMATRIX__

#include <assert.h>

#include <iostream>
#include <functional>

#include "types.hpp"
#include "alg.hpp"

 /* Matrix contains 3 templated properies:
  * T: The type of each cell
  * R, C: The number of rows and number of columns respectivly
  * Has support for basic matrix operations
  */
template <typename T = double>
class VMatrix {

public:
	const uint ROW_LENGTH;
	const uint COLUMN_LENGTH;
	const uint LENGTH;

private:
	/* Data is stored on the stack if VMatrix is initialised as a 2D heap array
	 *
	 * Translation to get index i from (x, y), where x is how far to the right, and y is how far down:
	 * F: (x,y) -> i : y * C + x
	 *
	 * Inverse translation, index to (x,y):
	 * F^-1: i -> (x,y) : (i / C, i % c)
	 */
	T* data = nullptr;

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
		assert(X < ROW_LENGTH && Y < COLUMN_LENGTH && "Attempt to index outside of matrix range");
		data[Y * ROW_LENGTH + X] = value;
	}

	// Sets data directly
	void set(T* data, uint length) {
		assert(length <= LENGTH && "");
		alg::copy(data, this->data, length);
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
		assert(X < ROW_LENGTH && Y < COLUMN_LENGTH && "Attempt to index outside of matrix range");
		return data[Y * ROW_LENGTH + X];
	}

	// gets data directly
	T* get() const {
		return data;
	}

	// Frees VMatrix
	~VMatrix() {
		free(data);
		data = nullptr;
	}

	// Constructor via reference
	VMatrix(const VMatrix& ref)
		: ROW_LENGTH(ref.ROW_LENGTH), COLUMN_LENGTH(ref.COLUMN_LENGTH), LENGTH(ROW_LENGTH * COLUMN_LENGTH) {

		data = (T*)malloc(sizeof(T) * LENGTH);

		alg::copy(ref.get(), data, LENGTH);
	}

	// Generates identity matrix
	VMatrix(int ROW_LENGTH, int COLUMN_LENGTH)
		: ROW_LENGTH(ROW_LENGTH), COLUMN_LENGTH(COLUMN_LENGTH), LENGTH(ROW_LENGTH * COLUMN_LENGTH) {
		assert(ROW_LENGTH == COLUMN_LENGTH && "Identity matrix only supported for square matrix");

		data = (T*)malloc(sizeof(T) * LENGTH);

		alg::fill(data, LENGTH, T(0));
		for (int i = 0; i < ROW_LENGTH; i++) {
			set(i, i, T(1));
		}
	}

	// Generates matrix and fills values
	VMatrix(int ROW_LENGTH, int COLUMN_LENGTH, T value)
		: ROW_LENGTH(ROW_LENGTH), COLUMN_LENGTH(COLUMN_LENGTH), LENGTH(ROW_LENGTH * COLUMN_LENGTH) {

		data = (T*)malloc(sizeof(T) * LENGTH);

		alg::fill(data, LENGTH, value);
	}

	// Generates values for matrix with a lambda f, mapping row, width to a value
	// f : (x:int, y:int). -> v:T
	VMatrix(int ROW_LENGTH, int COLUMN_LENGTH, std::function<T(uint, uint)> generator)
		: ROW_LENGTH(ROW_LENGTH), COLUMN_LENGTH(COLUMN_LENGTH), LENGTH(ROW_LENGTH * COLUMN_LENGTH) {
		for (uint i = 0; i < ROW_LENGTH; i++) {
			for (uint j = 0; j < COLUMN_LENGTH; j++) {
				set(i, j, generator(i, j));
			}
		}
	}

	// Conducts VMatrix element wise addition
	VMatrix operator+(const VMatrix& b) const {
		// Assert matrix dimensions are the same
		assert(this->ROW_LENGTH == b.ROW_LENGTH && this->COLUMN_LENGTH == b.COLUMN_LENGTH && "Matrix addition requries the same dimensions");
		
		// Create VMatrix to use as return
		VMatrix c(b);

		for (uint i = 0; i < LENGTH; i++) {
			c.data[i] += data[i];
		}

		return c;
	}

	// Conducts VMatrix addition against scalar
	VMatrix operator+(const T& b) const {
		// Create VMatrix to use as return
		VMatrix c(*this);

		for (uint i = 0; i < LENGTH; i++) {
			c.data[i] += b;
		}

		return c;
	}

	// Conducts VMatrix matrix multiplication in the form AB
	VMatrix operator*(const VMatrix& b) const {
		// Assert matrix dimensions for multiplication 
		assert(this->ROW_LENGTH == b.COLUMN_LENGTH  && "Matrix multiplication requries a.ROW_LENGTH == b.COLUMN_LENGTH");
		
		// Create VMatrix to use as return
		VMatrix c(b.ROW_LENGTH, this->COLUMN_LENGTH, T(0));

		for (uint i = 0; i < c.ROW_LENGTH; i++) {
			for (uint j = 0; j < c.COLUMN_LENGTH; j++) {
				T sum = T(0);
				for (uint p = 0; p < this->ROW_LENGTH; p++) {
					sum += this->get(p, j) * b.get(i, p);
				}
				c.set(i, j, sum);
			}
		}

		return c;
	}

	// Aplies a lambda F to each element in the function
	// F : T -> T for each cell in matrix
	VMatrix apply(T(*func)(T)) {
		VMatrix c(*this);

		for (uint i = 0; i < LENGTH; i++) {
			c.data[i] = func(c.data[i]);
		}

		return c;
	}
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const VMatrix<T>& m)
{
	for (uint j = 0; j < m.COLUMN_LENGTH; j++) {
		for (uint i = 0; i < m.ROW_LENGTH; i++) {
			os << m.get(i, j) << " ";
		}
		os << std::endl;
	}
	return os;
}

#endif