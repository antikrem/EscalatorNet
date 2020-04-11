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

private:
	uint rowLength;
	uint columnLength;
	uint length;

public:
	uint getRowLength() const {
		return rowLength;
	}

	uint getColumnLength() const {
		return columnLength;
	}

	uint getLength() const {
		return length;
	}

private:
	/* Data is stored on the heap  as a 2D array
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
		assert(x < rowLength && y < columnLength && "Attempt to index outside of matrix range");
		data[y * rowLength + x] = value;
	}

	// Set value by (x,y) coord
	// is templated
	template<uint X, uint Y>
	void set(T value) {
		assert(X < rowLength && Y < columnLength && "Attempt to index outside of matrix range");
		data[Y * rowLength + X] = value;
	}

	// Sets data directly
	void set(T* data, uint length) {
		assert(length <= length && "");
		alg::copy(data, this->data, length);
	}

	// get value by (x,y) coord
	// Throws assertion error on invalid x,y in debug
	T get(uint x, uint y) const {
		assert(x < rowLength && y < columnLength && "Attempt to index outside of matrix range");
		return data[y * rowLength + x];
	}

	// get value by (x,y) coord
	// is templated
	template<uint X, uint Y>
	T get() const {
		assert(X < rowLength && Y < columnLength && "Attempt to index outside of matrix range");
		return data[Y * rowLength + X];
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
		: rowLength(ref.rowLength), columnLength(ref.columnLength), length(rowLength * columnLength) {

		data = (T*)malloc(sizeof(T) * length);

		alg::copy(ref.get(), data, length);
	}

	// Generates identity matrix
	VMatrix(uint rowLength, uint columnLength)
		: rowLength(rowLength), columnLength(columnLength), length(rowLength * columnLength) {
		assert(rowLength == columnLength && "Identity matrix only supported for square matrix");

		data = (T*)malloc(sizeof(T) * length);

		alg::fill(data, length, T(0));
		for (int i = 0; i < rowLength; i++) {
			set(i, i, T(1));
		}
	}

	// Generates matrix and fills values
	VMatrix(uint rowLength, uint columnLength, T value)
		: rowLength(rowLength), columnLength(columnLength), length(rowLength * columnLength) {

		data = (T*)malloc(sizeof(T) * length);

		alg::fill(data, length, value);
	}

	// Generates values for matrix with a lambda f, mapping row, width to a value
	// f : (x:int, y:int). -> v:T
	VMatrix(uint rowLength, uint columnLength, std::function<T(uint, uint)> generator)
		: rowLength(rowLength), columnLength(columnLength), length(rowLength * columnLength) {
		
		data = (T*)malloc(sizeof(T) * length);

		for (uint i = 0; i < rowLength; i++) {
			for (uint j = 0; j < columnLength; j++) {
				set(i, j, generator(i, j));
			}
		}
	}

	// Fixed size assignment operator
	VMatrix& operator=(const VMatrix& b) {
		// Assert matrix dimensions are the same
		assert(this->rowLength == b.rowLength && this->columnLength == b.columnLength && "Matrix safe assignment requries the same dimensions");

		for (uint i = 0; i < length; i++) {
			this->data[i] += b.data[i];
		}

		return *this;
	}

	// Assigns this VMatrix to be an exact deep copy of another
	// No restriction on size of matrix
	void assign(VMatrix& b) {
		this->rowLength = b.rowLength;
		this->columnLength = b.columnLength;
		this->length = b.length;

		free(data);
		data = (T*)malloc(sizeof(T) * length);

		alg::copy(this->data, b.data, length);

	}

	// Conducts VMatrix element wise addition
	VMatrix operator+(const VMatrix& b) const {
		// Assert matrix dimensions are the same
		assert(this->rowLength == b.rowLength && this->columnLength == b.columnLength && "Matrix addition requries the same dimensions");
		
		// Create VMatrix to use as return
		VMatrix c(b);

		for (uint i = 0; i < length; i++) {
			c.data[i] += data[i];
		}

		return c;
	}

	// Conducts VMatrix addition against scalar
	VMatrix operator+(const T& b) const {
		// Create VMatrix to use as return
		VMatrix c(*this);

		for (uint i = 0; i < length; i++) {
			c.data[i] += b;
		}

		return c;
	}

	// Conducts VMatrix matrix multiplication in the form AB
	VMatrix operator*(const VMatrix& b) const {
		// Assert matrix dimensions for multiplication 
		assert(this->rowLength == b.columnLength  && "Matrix multiplication requries a.ROW_LENGTH == b.COLUMN_LENGTH");
		
		// Create VMatrix to use as return
		VMatrix c(b.rowLength, this->columnLength, T(0));

		for (uint i = 0; i < c.rowLength; i++) {
			for (uint j = 0; j < c.columnLength; j++) {
				T sum = T(0);
				for (uint p = 0; p < this->rowLength; p++) {
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

		for (uint i = 0; i < length; i++) {
			c.data[i] = func(c.data[i]);
		}

		return c;
	}
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const VMatrix<T>& m)
{
	for (uint j = 0; j < m.getColumnLength(); j++) {
		for (uint i = 0; i < m.getRowLength(); i++) {
			os << m.get(i, j) << " ";
		}
		os << std::endl;
	}
	return os;
}

#endif