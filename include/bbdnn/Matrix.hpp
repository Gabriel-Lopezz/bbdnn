#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <math.h>
#include <random>
#include <initializer_list>

namespace bbdnn {

    struct Vector;

    /// Lightweight dense matrix of floats.
    class Matrix {
    private:
        int rows;
        int cols;
        int elementCount;
    
        float* data;

        void destroyMatrixData();

    public:
        /// Create an empty matrix (0x0).
        Matrix();
        /// Create a matrix with given rows and columns.
        Matrix(int Rows, int Cols);
        /// Create a matrix filled with a default value.
        Matrix(int Rows, int Cols, float defaultVal);
        /// Create a matrix from a raw data array.
        explicit Matrix(int Rows, int Cols, float Data[]);
        /// Copy-construct from another matrix.
        Matrix(const Matrix& other);
        /// Destroy the matrix and free storage.
        ~Matrix();

        /// Assign from another matrix (deep copy).
        Matrix& operator=(const Matrix& other);
        /// Matrix multiplication.
        Matrix operator*(const Matrix& other) const;
        /// Multiply by scalar.
        Matrix operator*(float scalar) const;
        /// Element-wise addition.
        Matrix operator+(const Matrix& other) const;
        /// Element-wise subtraction.
        Matrix operator-(const Matrix& other) const;
        /// Divide by scalar.
        Matrix operator/(float scalar) const;
        /// In-place element-wise addition.
        Matrix& operator+=(const Matrix& other);
        /// In-place scalar multiplication.
        Matrix& operator*=(float scalar);
        /// In-place scalar division.
        Matrix& operator/=(float scalar);
        /// Get a mutable pointer to a row.
        float* operator[](int ind);
        /// Access an element by row and column (mutable).
        float& operator()(int row, int col);
        /// Access an element by row and column (const).
        const float& operator()(int row, int col) const;
    
        /// Number of rows.
        int Rows() const;
        /// Number of columns.
        int Cols() const;
        /// Total element count.
        int size() const;
        /// Return a newly allocated column array (caller owns).
        float* getCol(int col) const;
        /// Return a newly allocated row array (caller owns).
        float* getRow(int col) const;
        /// Bounds-checked element access (mutable).
        float& at(int row, int col);
        /// Bounds-checked element access (const).
        const float& at(int row, int col) const;

        /// Sum of all elements.
        float sum() const;
        /// Return the transposed matrix.
        Matrix transposed() const;
        /// Apply matrix to a column-vector input.
        Vector applyMatrix(const Matrix& inputs) const;
        /// Element-wise product.
        Matrix hadamardProduct(const Matrix& other) const;
    
        /// Xavier initializer.
        static Matrix xavierMatrix(int inCount, int outCount, uint_fast32_t randomSeed);
        /// Kaiming initializer.
        static Matrix kaimingMatrix(int inCount, int outCount, uint_fast32_t randomSeed);

        /// Print matrix to stderr.
        void printMatrix() const;
    };

/// Column vector (Nx1) convenience type.
class Vector : public Matrix {
    public:
        /// Construct an empty vector.
        Vector();
        /// Construct from a 1-column matrix.
        Vector(const Matrix& other);
        /// Construct a vector with size and default value.
        Vector(int Size, float defaultVal = 0.0f);
        /// Construct from a raw array.
        Vector(float vals[], int Size);
        /// Construct from an initializer list.
        Vector(std::initializer_list<float> vals);

        /// Assign from a 1-column matrix.
        Vector& operator=(const Matrix& other);
        /// Element access (mutable).
        float& operator[](int ind);
        /// Element access (const).
        const float& operator[](int ind) const;
    };

}

#endif
