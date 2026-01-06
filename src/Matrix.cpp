#include "bbdnn/Matrix.hpp"
#include <stdexcept>

namespace bbdnn {

    Matrix::Matrix() : rows(0), cols(0), elementCount(0), data(nullptr) { }

    Matrix::Matrix(int Rows, int Cols) : rows(Rows), cols(Cols), elementCount(Rows * Cols) {
        data = new float[elementCount];
    }

    Matrix::Matrix(int Rows, int Cols, float defaultVal) : rows(Rows), cols(Cols), elementCount(Rows * Cols) {
        data = new float[elementCount];

        for (int i = 0; i < elementCount; i++)
            data[i] = defaultVal;
    }

    Matrix::Matrix(int Rows, int Cols, float Data[]) : rows(Rows), cols(Cols), elementCount(Rows * Cols), data(Data) { }

    Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), elementCount(other.elementCount) {
        data = new float[elementCount];

        for (int i = 0; i < elementCount; i++)
            data[i] = other.data[i];
    }

    Matrix::~Matrix() {
        if (data == nullptr)
            return;

        destroyMatrixData();
    }

    void Matrix::destroyMatrixData() {
        delete[] data;
        data = nullptr;
    }

    Matrix& Matrix::operator=(const Matrix& other) {
        if (this == &other)
            return *this;
    
        destroyMatrixData();

        rows = other.rows;
        cols = other.cols;
        elementCount = other.elementCount;

        data = new float[elementCount];

        for (int i = 0; i < elementCount; i++)
            data[i] = other.data[i];

        return *this;
    }

    Matrix Matrix::operator*(const Matrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("Matrix column does not match other's row.");

        Matrix product(rows, other.cols, 0.0f);
    
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < other.cols; c++)
                for (int i = 0; i < cols; i++)
                    product.at(r,c) += this->at(r, i) * other.at(i, c);

        return product;
    }

    Matrix Matrix::operator*(float scalar) const {
        Matrix product(rows, cols);
    
        for (int i = 0; i < elementCount; i++)
            product.data[i] = data[i] * scalar;

        return product;
    }

    Matrix Matrix::operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrices must be of the same dimensions");

        Matrix result(rows, cols);
    
        for (int i = 0; i < elementCount; i++)
            result.data[i] = data[i] + other.data[i];
        
        return result;
    }

    Matrix Matrix::operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrices must be of the same dimensions");

        Matrix result(rows, cols);
    
        for (int i = 0; i < elementCount; i++) {
            result.data[i] = data[i] - other.data[i];
        }
    
        return result;
    }

    Matrix Matrix::operator/(float scalar) const {
        Matrix quotient(rows, cols);
    
        for (int i = 0; i < elementCount; i++)
            quotient.data[i] = data[i] / scalar;

        return quotient;
    }

    Matrix& Matrix::operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("Matrices must be of the same dimensions");

        for (int i = 0; i < elementCount; i++)
            data[i] += other.data[i];

        return *this;
    }

    Matrix& Matrix::operator*=(float scalar) {
        for (int i = 0; i < elementCount; i++)
            data[i] *= scalar;

        return *this;
    }

    Matrix& Matrix::operator/=(float scalar) {
        for (int i = 0; i < elementCount; i++)
            data[i] /= scalar;

        return *this;
    }

    float* Matrix::operator[](int row) {
        int offset = row * cols;
        float* rowAddress = data + offset;
    
        return rowAddress;
    }

    const float& Matrix::operator()(int row, int col) const {
        return at(row, col);
    }

    float& Matrix::operator()(int row, int col) {
        return at(row, col);
    }

    float Matrix::sum() const {
        float acc = 0;

        for (int i = 0; i < elementCount; i++)
        {
            acc += data[i];
        }
    
        return acc;
    }

    Matrix Matrix::transposed() const {
        Matrix m(cols, rows);

        for (int i = 0; i < m.rows; i++)
            for (int j = 0; j < m.cols; j++)
                m.at(i, j) = this->at(j, i);

        return m;
    }

    Matrix Matrix::hadamardProduct(const Matrix& other) const {
        if (other.rows != rows || other.cols != cols)
            throw std::invalid_argument("Hadamard Product components must be of equal dimensions");
    
        Matrix result(rows, cols);

        for (int i = 0; i < elementCount; i++)
            result.data[i] = other.data[i] * data[i];
    
        return result;
    }

    Vector Matrix::applyMatrix(const Matrix& inputs) const {
        if (inputs.Cols() != 1)
            throw std::invalid_argument("Input matrix must be a column vector");

        if (inputs.Rows() != rows)
            throw std::invalid_argument("Input Length does not match Matrix rows");

        Vector res(cols, 0.0f);

        for (int i = 0; i < rows; i++)
            for (int c = 0; c < cols; c++)
                res[c] += inputs.at(i, 0) * at(i, c);

        return res;
    }

    int Matrix::size() const {
        return elementCount;
    }

    int Matrix::Rows() const {
        return rows;
    }

    int Matrix::Cols() const {
        return cols;
    }

    float* Matrix::getRow(int row) const {
        float* rowData = new float[cols];

        for (int i = 0; i < cols; i++)
            rowData[i] = at(row, i);
    
        return rowData;
    }

    float* Matrix::getCol(int col) const {
        float* column = new float[rows];

        for (int i = 0; i < rows; i++)
            column[i] = at(i, col);
    
        return column;
    }

    float& Matrix::at(int row, int col) {
        int offset = row * cols + col;
        // FOR DEBUGGING PURPOSES
        if (offset >= elementCount || offset < 0)
        {
            std::cerr << "Invalid row/col | Row: " << row << " Col: " << col << "| rows(): " << rows << " cols(): " << cols << std::endl << std::flush;
            throw std::runtime_error("Out of bounds");
        }
    
        float& value = data[offset];

        return value;
    }

    const float& Matrix::at(int row, int col) const {
            int offset = row * cols + col;
        // FOR DEBUGGING PURPOSES
        if (offset >= elementCount || offset < 0)
        {
            std::cerr << "Invalid row/col | Row: " << row << " Col: " << col << "| rows(): " << rows << " cols(): " << cols << std::endl << std::flush;
            throw std::runtime_error("Out of bounds");
        }
    
        const float& value = data[offset];

        return value;
    }

    Matrix Matrix::xavierMatrix(int inCount, int outCount, uint_fast32_t randomSeed) {
        float endpoint = sqrt(6.0f / float(inCount + outCount));

        std::mt19937 generator(randomSeed);
        std::uniform_real_distribution<float> distribution(-endpoint, endpoint);

        Matrix result(inCount, outCount);
    
        for(int i = 0; i < result.rows; i++)
            for(int j = 0; j < result.cols; j++)
                result.at(i, j) = distribution(generator);

        return result;
    }

    Matrix Matrix::kaimingMatrix(int inCount, int outCount, uint_fast32_t randomSeed) {
        float std = sqrt(2.0f / float(inCount));

        std::mt19937 generator(randomSeed);
        std::normal_distribution<float> distribution(0.0f, std);

        Matrix result(inCount, outCount);
    
        for(int i = 0; i < result.rows; i++)
            for(int j = 0; j < result.cols; j++)
                result.at(i, j) = distribution(generator);
    
        return result;
    }

    void Matrix::printMatrix() const {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                std::cerr << at(i, j) << " ";

            std::cerr << std::endl;
        }
    }

    Vector::Vector() : Matrix() { }

    Vector::Vector(const Matrix& other) : Matrix(other) {
        if (other.Cols() != 1)
            throw std::invalid_argument("Vector objects must have only 1 col.");
    }

    Vector::Vector(int Size, float defaultVal) : Matrix(Size, 1, defaultVal) {}

    Vector::Vector(float vals[], int Size): Matrix(Size, 1) {
        for (int i = 0; i < Size; i++)
            (*this)[i] = vals[i];
    }

    Vector::Vector(std::initializer_list<float> vals) : Matrix(vals.size(), 1) {
        int i = 0;
        for (float val : vals) {
            (*this)[i] = val;
            i++;
        }
    }

    Vector& Vector::operator=(const Matrix& other) {
        if (other.Cols() != 1)
            throw std::invalid_argument("Vector objects must have only 1 row.");
    
        Matrix::operator=(other);

        return *this;
    }

    float& Vector::operator[](int ind) {
        return at(ind, 0);
    }

    const float& Vector::operator[](int ind) const {
        return at(ind, 0);
    }

}
