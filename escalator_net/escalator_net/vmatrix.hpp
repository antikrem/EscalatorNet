/* Runtime implementation of matrix
 * is statically generated
 * Has runtime specified size
 */

#ifndef __VMATRIX__
#define __VMATRIX__

#include <assert.h>

#include <iostream>
#include <functional>

#include <vector>

#include "types.hpp"
#include "alg.hpp"

 /* Implementation of a variable matrix
  * Has run-time specified dimensions
  * T: element type
  */
template <typename T = double>
class VMatrix {

private:
	// Dimensions of matrix, internally managed
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
	/* Data is stored on the heap as a 2D array
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
		for (uint i = 0; i < rowLength; i++) {
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

	// Creates a VMatrix from a vector of vectors
	// Each sub vector is a row
	// NO INPUT CHECKING ON VECTOR LENGTHS (TODO)
	VMatrix(std::vector<std::vector<T>> input)
		: rowLength(input[0].size()), columnLength(input.size()), length(rowLength * columnLength) {

		data = (T*)malloc(sizeof(T) * length);

		for (uint j = 0; j < columnLength; j++) {
			// copy in all data from vector
			std::copy(input[j].data(), input[j].data() + rowLength, data + j * rowLength);
		}
		
	}

	// Extends this VMatrix by a single/many rows
	// New rows come in as VMatrix of the same row size
	void extend(const VMatrix<T>& input) {
		assert(input.rowLength == rowLength && "Extending requires same row lengths");

		columnLength += input.columnLength;
		// remember where to start for extension to internal data
		uint offset = length ;
		length = columnLength * rowLength;

		// extend data range
		data = (T*)realloc(data, sizeof(T) * length);

		// copy over data to end
		alg::copy(input.data, data + offset, input.length);
	}

	// Returns a matrix 
	VMatrix<T> getColumn(uint row) const {
		VMatrix<T> c(1, this->columnLength, T(0.0));

		for (uint i = 0; i < this->columnLength; i++) {
			c.set(0, i, this->get(row, i));
		}

		return c;
	}

	// Fixed size assignment operator
	VMatrix& operator=(const VMatrix& b) {
		// Assert matrix dimensions are the same
		assert(this->rowLength == b.rowLength && this->columnLength == b.columnLength && "Matrix safe assignment requries the same dimensions");

		alg::copy(b.data, this->data, length);

		return *this;
	}

	// Assigns this VMatrix to be an exact deep copy of another
	// No restriction on size of matrix
	// This matrix will change size to acommodate
	void assign(const VMatrix& b) {
		this->rowLength = b.rowLength;
		this->columnLength = b.columnLength;
		this->length = b.length;

		free(data);
		data = (T*)malloc(sizeof(T) * length);

		alg::copy(b.data, this->data, length);

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

	// Conducts VMatrix element wise substraction
	VMatrix operator-(const VMatrix& b) const {
		// Assert matrix dimensions are the same
		assert(this->rowLength == b.rowLength && this->columnLength == b.columnLength && "Matrix subtraction requries the same dimensions");

		// Create VMatrix to use as return
		VMatrix c(*this);

		for (uint i = 0; i < length; i++) {
			c.data[i] -= b.data[i];
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

	// Conducts VMatrix scalar multiplication in the form Ak
	VMatrix operator*(const T& k) const {
		// Assert matrix dimensions for multiplication 
		
		VMatrix c(*this);

		for (uint i = 0; i < length; i++) {
			c.data[i] *= k;
		}

		return c;
	}

	// Conducts VMatrix elementwise multiplication
	VMatrix elementMultiply(const VMatrix& b) const {
		// Assert matrix dimensions for multiplication 
		assert(this->rowLength == b.rowLength && this->columnLength == b.columnLength
			&& "Matrix elementwise multiplication requries same size matrices");

		// Create VMatrix to use as return
		VMatrix c(this->rowLength, this->columnLength, T(0));

		for (uint i = 0; i < c.rowLength; i++) {
			for (uint j = 0; j < c.columnLength; j++) {
				c.set(i, j, this->get(i, j) * b.get(i, j));
			}
		}

		return c;
	}

	// Aplies a lambda F to each element in the function
	// F : T -> T for each cell in matrix
	VMatrix apply(T(*func)(T)) const {
		VMatrix c(*this);
		for (uint i = 0; i < length; i++) {
			c.data[i] = func(c.data[i]);
		}

		return c;
	}

	// Slow transpose
	VMatrix<T> transpose() const {
		VMatrix<T> c(this->columnLength, this->rowLength, T(0.0f));

		for (uint i = 0; i < rowLength; i++) {
			for (uint j = 0; j < columnLength; j++) {
				c.set(j, i, get(i, j));
			}
		}

		return c;
	}
	
	// Fast transpose, only works with single vectors
	VMatrix<T> qTranspose() const {
		assert(rowLength == 1 || columnLength == 1 && "qTransport only supported on vectors");
		VMatrix<T> c(*this);

		std::swap(c.rowLength, c.columnLength);

		return c;
	}

	// Sums all values in the matrix, and returns sum
	T sum() const {
		T sum = T(0);
		for (uint i = 0; i < length; i++) {
			sum += data[i];
		}
		return sum;
	}

	// Sums all values in a collumn and returns VMatrix with column length 1
	VMatrix<T> sumColumns() const {
		VMatrix<T> c(rowLength, 1, T(0));

		for (uint i = 0; i < rowLength; i++) {
			T v = T(0);
			for (uint j = 0; j < columnLength; j++) {
				v += get(i, j);
			}
			c.set(i, 0, v);
		}

		return c;
	}

	// Sums all values in a collumn and returns VMatrix with row length 1
	VMatrix<T> sumRows() const {
		VMatrix<T> c(1, columnLength, T(0));

		for (uint j = 0; j < columnLength; j++) {
			T v = T(0);
			for (uint i = 0; i < rowLength; i++) {
				v += get(i, j);
			}
			c.set(0, j, v);
		}

		return c;
	}

	// Returns element that has the maximum value
	T max() const {
		T uBound = data[0];
		for (uint i = 1; i < length; i++) {
			uBound = data[i] > uBound ? data[i] : uBound;
		}
		return uBound;
	}

	// Returns element that has the minimum value
	T min() const {
		T lBound = data[0];
		for (uint i = 1; i < length; i++) {
			lBound = data[i] < lBound ? data[i] : lBound;
		}
		return lBound;
	}

	// Clamps all elements to given range
	void clamp(T lower, T upper) {
		for (uint i = 0; i < length; i++) {
			data[i] = std::min(upper, std::max(lower, data[i]));
		}
	}
};

template <typename T>
static std::ostream& operator<<(std::ostream& os, const VMatrix<T>& m)
{
	for (uint j = 0; j < m.getColumnLength(); j++) {
		// Only print after first line
		if (j) {
			os << std::endl;
		}

		for (uint i = 0; i < m.getRowLength(); i++) {
			os << m.get(i, j) << " ";
		}
		
	}
	return os;
}

#endif