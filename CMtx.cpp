#include "CMtx.h"
#include <stdlib.h>
#include <iostream>
#include <immintrin.h>
#include <thread>
#include <exception>


// global constans
const float MyAlgebra::CMtx::ALG_PRECISION = 0.000001f;
const int MyAlgebra::CMtx::N_JOBS = 4;
const int MyAlgebra::CMtx::VECTOR_LENGTH = 8;
// SIMD parameters
const int MyAlgebra::CMtx::ADD_SIZE = VECTOR_LENGTH * 6;
const int MyAlgebra::CMtx::SUB_SIZE = VECTOR_LENGTH * 6;
const int MyAlgebra::CMtx::MULT_SIZE = VECTOR_LENGTH * 3;
const int MyAlgebra::CMtx::SCALAR_MULT_SIZE = VECTOR_LENGTH * 6;
// thresholds
const int MyAlgebra::CMtx::THRESHOLD_MUL = 576;

// CONSTRUCTORS

// private constructor
MyAlgebra::CMtx::CMtx(uint16_t rows, uint16_t cols, float* matrix) :
	rows(rows),
	cols(cols),
	elements(rows* cols),
	matrix(matrix)
{}

// standard
MyAlgebra::CMtx::CMtx(uint16_t rows, uint16_t cols, bool rand_init) :
	rows(rows),
	cols(cols),
	elements(rows* cols)
{
	matrix = new float[elements];
	if (rand_init)
	{
		for (int i = 0; i < elements; i++)
		{
			matrix[i] = (static_cast<float> (rand()) / RAND_MAX);
		}
	}
}

// diagonal
MyAlgebra::CMtx::CMtx(uint16_t rows, float diagonal) :
	rows(rows),
	cols(rows),
	elements(rows* rows)
{
	matrix = new float[elements];
	int diagonal_idx = 0;
	for (int i = 0; i < elements; i++)
	{
		if (i == diagonal_idx)
		{
			matrix[i] = diagonal;
			diagonal_idx += rows + 1;
		}
		else
		{
			matrix[i] = .0f;
		}
	}
}

// copy
MyAlgebra::CMtx::CMtx(const CMtx& other) :
	rows(other.rows),
	cols(other.cols),
	elements(other.elements)
{
	matrix = new float[other.elements];
	for (int i = 0; i < elements; i++)
	{
		matrix[i] = other.matrix[i];
	}

}

// move
MyAlgebra::CMtx::CMtx(CMtx&& other) :
	rows(other.rows),
	cols(other.cols),
	elements(other.elements),
	matrix(other.matrix)
{
	other.matrix = nullptr;
	other.rows = 0;
	other.cols = 0;
	other.elements = 0;
}

// destructor
MyAlgebra::CMtx::~CMtx()
{
	delete[] matrix;
}

// ASSIGNMENTS OPERATORS

// copy
const MyAlgebra::CMtx& MyAlgebra::CMtx::operator=(const CMtx& other)
{
	if (&other != this)
	{
		for (int i = 0; i < other.elements; i++)
		{
			matrix[i] = other.matrix[i];
		}
		rows = other.rows;
		cols = other.cols;
		elements = other.elements;
	}
	return *this;
}

// diagonal
const MyAlgebra::CMtx& MyAlgebra::CMtx::operator=(float diagonal)
{
	if (rows != cols)
	{
		throw std::exception("Cannot assign diagonal matrix to non-square matrix.");
	}
	int diagonal_idx = 0;
	for (int i = 0; i < elements; i++)
	{
		if (i == diagonal_idx)
		{
			matrix[i] = diagonal;
			diagonal_idx += rows + 1;
		}
		else
		{
			matrix[i] = .0f;
		}
	}
	return *this;
}

// move
const MyAlgebra::CMtx& MyAlgebra::CMtx::operator=(CMtx&& other)
{
	delete[] matrix;
	matrix = other.matrix;
	rows = other.rows;
	cols = other.cols;
	elements = other.elements;

	other.matrix = nullptr;
	other.rows = 0;
	other.cols = 0;
	other.elements = 0;
	return *this;
}

// GET ITEM

float* MyAlgebra::CMtx::operator[](int idx)
{
	if ((rows < idx) || (idx < 0))
	{
		throw std::exception("Invalid index.");
	}
	return &matrix[cols * idx];
}

// MATRIX MULTIPLICATION

MyAlgebra::CMtx MyAlgebra::CMtx::operator*(const CMtx& other) const&
{
	if (cols != other.rows)
	{
		throw std::exception("Invalid matrices shape");
	}
	int dot_elements = rows * other.cols;
	float* new_matrix = new float[dot_elements];
	std::fill_n(new_matrix, dot_elements, .0f);
	if (elements < THRESHOLD_MUL)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				for (int k = 0; k < other.cols; k++)
				{
					new_matrix[i * other.cols + k] += matrix[i * cols + j] * other.matrix[j * other.cols + k];
				}
			}
		}
		return CMtx(rows, other.cols, new_matrix);
	}
	else
	{
		std::thread jobs[N_JOBS];
		CMtx mtx = ~*this;
		for (int i = 0; i < N_JOBS; i++)
		{
			jobs[i] = std::thread(
				&MyAlgebra::CMtx::matmul,
				this,
				i * rows / N_JOBS,
				(i + 1) * rows / N_JOBS,
				std::ref(mtx),
				std::ref(other),
				std::ref(new_matrix)
			);
		}
		for (int i = 0; i < N_JOBS; i++)
		{
			jobs[i].join();
		}
		return CMtx(rows, other.cols, new_matrix);
	}
}

void MyAlgebra::CMtx::matmul(int idx0, int idx1, const CMtx& this_transposed, const CMtx& other, float* result) const
{
	// this transposed
	__m256 _t1, _t2, _t3;
	// other
	__m256 _o1, _o2, _o3;
	// store
	__m256 _s1, _s2, _s3, _s4, _s5, _s6, _s7, _s8, _s9;
	// indexes this transposed
	int pt1, pt2, pt3;
	// indexes other 
	int po1, po2, po3;
	// indexes store
	int ps1, ps2, ps3, ps4, ps5, ps6, ps7, ps8, ps9;

	for (int i = 0; i < cols; i++)
	{
		int j = idx0;

		for (; j < idx1 - 3; j += 3)
		{
			int k = 0;
			for (; k < other.cols - MULT_SIZE; k += MULT_SIZE)
			{
				pt1 = i * rows + j;
				pt2 = pt1 + 1;
				pt3 = pt2 + 1;

				po1 = i * other.cols + k;
				po2 = po1 + VECTOR_LENGTH;
				po3 = po2 + VECTOR_LENGTH;

				ps1 = j * other.cols + k;
				ps2 = ps1 + VECTOR_LENGTH;
				ps3 = ps2 + VECTOR_LENGTH;

				ps4 = (j + 1) * other.cols + k;
				ps5 = ps4 + VECTOR_LENGTH;
				ps6 = ps5 + VECTOR_LENGTH;

				ps7 = (j + 2) * other.cols + k;
				ps8 = ps7 + VECTOR_LENGTH;
				ps9 = ps8 + VECTOR_LENGTH;

				_t1 = _mm256_set1_ps(this_transposed.matrix[pt1]);
				_t2 = _mm256_set1_ps(this_transposed.matrix[pt2]);
				_t3 = _mm256_set1_ps(this_transposed.matrix[pt3]);

				_o1 = _mm256_loadu_ps(&other.matrix[po1]);
				_o2 = _mm256_loadu_ps(&other.matrix[po2]);
				_o3 = _mm256_loadu_ps(&other.matrix[po3]);

				_s1 = _mm256_mul_ps(_t1, _o1);
				_s2 = _mm256_mul_ps(_t1, _o2);
				_s3 = _mm256_mul_ps(_t1, _o3);

				_s4 = _mm256_mul_ps(_t2, _o1);
				_s5 = _mm256_mul_ps(_t2, _o2);
				_s6 = _mm256_mul_ps(_t2, _o3);

				_s7 = _mm256_mul_ps(_t3, _o1);
				_s8 = _mm256_mul_ps(_t3, _o2);
				_s9 = _mm256_mul_ps(_t3, _o3);


				_mm256_storeu_ps(&result[ps1], _mm256_add_ps(_mm256_loadu_ps(&result[ps1]), _s1));
				_mm256_storeu_ps(&result[ps2], _mm256_add_ps(_mm256_loadu_ps(&result[ps2]), _s2));
				_mm256_storeu_ps(&result[ps3], _mm256_add_ps(_mm256_loadu_ps(&result[ps3]), _s3));

				_mm256_storeu_ps(&result[ps4], _mm256_add_ps(_mm256_loadu_ps(&result[ps4]), _s4));
				_mm256_storeu_ps(&result[ps5], _mm256_add_ps(_mm256_loadu_ps(&result[ps5]), _s5));
				_mm256_storeu_ps(&result[ps6], _mm256_add_ps(_mm256_loadu_ps(&result[ps6]), _s6));

				_mm256_storeu_ps(&result[ps7], _mm256_add_ps(_mm256_loadu_ps(&result[ps7]), _s7));
				_mm256_storeu_ps(&result[ps8], _mm256_add_ps(_mm256_loadu_ps(&result[ps8]), _s8));
				_mm256_storeu_ps(&result[ps9], _mm256_add_ps(_mm256_loadu_ps(&result[ps9]), _s9));
			}

			for (; k < other.cols; ++k)
			{
				pt1 = i * rows + j;
				pt2 = i * rows + j + 1;
				pt3 = i * rows + j + 2;

				po1 = i * other.cols + k;

				ps1 = j * other.cols + k;
				ps2 = (j + 1) * other.cols + k;
				ps3 = (j + 2) * other.cols + k;

				result[ps1] += this_transposed.matrix[pt1] * other.matrix[po1];
				result[ps2] += this_transposed.matrix[pt2] * other.matrix[po1];
				result[ps3] += this_transposed.matrix[pt3] * other.matrix[po1];
			}
		}

		for (; j < idx1; ++j)
		{
			int k = 0;
			for (; k < other.cols - MULT_SIZE; k += MULT_SIZE)
			{
				pt1 = i * rows + j;

				po1 = i * other.cols + k;
				po2 = po1 + VECTOR_LENGTH;
				po3 = po2 + VECTOR_LENGTH;
				
				ps1 = j * other.cols + k;
				ps2 = ps1 + VECTOR_LENGTH;
				ps3 = ps2 + VECTOR_LENGTH;

				_t1 = _mm256_set1_ps(this_transposed.matrix[pt1]);
				_o1 = _mm256_loadu_ps(&other.matrix[po1]);
				_o2 = _mm256_loadu_ps(&other.matrix[po2]);
				_o3 = _mm256_loadu_ps(&other.matrix[po3]);

				_s1 = _mm256_mul_ps(_t1, _o1);
				_s2 = _mm256_mul_ps(_t1, _o2);
				_s3 = _mm256_mul_ps(_t1, _o3);

				_mm256_storeu_ps(&result[ps1], _mm256_add_ps(_mm256_loadu_ps(&result[ps1]), _s1));
				_mm256_storeu_ps(&result[ps2], _mm256_add_ps(_mm256_loadu_ps(&result[ps2]), _s2));
				_mm256_storeu_ps(&result[ps3], _mm256_add_ps(_mm256_loadu_ps(&result[ps3]), _s3));

			}
			for (; k < other.cols; ++k)
			{
				result[j * other.cols + k] +=
					this_transposed.matrix[i * rows + j] * other.matrix[i * other.cols + k];
			}
		}
	}
}

// MATRIX ADDITION

MyAlgebra::CMtx MyAlgebra::CMtx::operator+(const CMtx& other) const&
{
	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	float* new_matrix = new float[elements];
	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;


	int i = 0;
	for (; i < elements - ADD_SIZE; i += ADD_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_add_ps(_t1, _o1);
		_t2 = _mm256_add_ps(_t2, _o2);
		_t3 = _mm256_add_ps(_t3, _o3);
		_t4 = _mm256_add_ps(_t4, _o4);
		_t5 = _mm256_add_ps(_t5, _o5);
		_t6 = _mm256_add_ps(_t6, _o6);

		_mm256_storeu_ps(&new_matrix[pt1], _t1);
		_mm256_storeu_ps(&new_matrix[pt2], _t2);
		_mm256_storeu_ps(&new_matrix[pt3], _t3);
		_mm256_storeu_ps(&new_matrix[pt4], _t4);
		_mm256_storeu_ps(&new_matrix[pt5], _t5);
		_mm256_storeu_ps(&new_matrix[pt6], _t6);

		pt1 += ADD_SIZE;
		pt2 += ADD_SIZE;
		pt3 += ADD_SIZE;
		pt4 += ADD_SIZE;
		pt5 += ADD_SIZE;
		pt6 += ADD_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_add_ps(_t1, _o1);
		_mm256_storeu_ps(&new_matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		new_matrix[i] = matrix[i] + other.matrix[i];
	}
	return CMtx(rows, cols, new_matrix);
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator+(CMtx&& other) const&
{
	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;


	int i = 0;
	for (; i < elements - ADD_SIZE; i += ADD_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_add_ps(_t1, _o1);
		_t2 = _mm256_add_ps(_t2, _o2);
		_t3 = _mm256_add_ps(_t3, _o3);
		_t4 = _mm256_add_ps(_t4, _o4);
		_t5 = _mm256_add_ps(_t5, _o5);
		_t6 = _mm256_add_ps(_t6, _o6);

		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		_mm256_storeu_ps(&other.matrix[pt2], _t2);
		_mm256_storeu_ps(&other.matrix[pt3], _t3);
		_mm256_storeu_ps(&other.matrix[pt4], _t4);
		_mm256_storeu_ps(&other.matrix[pt5], _t5);
		_mm256_storeu_ps(&other.matrix[pt6], _t6);

		pt1 += ADD_SIZE;
		pt2 += ADD_SIZE;
		pt3 += ADD_SIZE;
		pt4 += ADD_SIZE;
		pt5 += ADD_SIZE;
		pt6 += ADD_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_add_ps(_t1, _o1);
		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		other.matrix[i] += matrix[i];
	}
	return other;
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator+(CMtx&& other)&&
{

	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;


	int i = 0;
	for (; i < elements - ADD_SIZE; i += ADD_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_add_ps(_t1, _o1);
		_t2 = _mm256_add_ps(_t2, _o2);
		_t3 = _mm256_add_ps(_t3, _o3);
		_t4 = _mm256_add_ps(_t4, _o4);
		_t5 = _mm256_add_ps(_t5, _o5);
		_t6 = _mm256_add_ps(_t6, _o6);

		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		_mm256_storeu_ps(&other.matrix[pt2], _t2);
		_mm256_storeu_ps(&other.matrix[pt3], _t3);
		_mm256_storeu_ps(&other.matrix[pt4], _t4);
		_mm256_storeu_ps(&other.matrix[pt5], _t5);
		_mm256_storeu_ps(&other.matrix[pt6], _t6);

		pt1 += ADD_SIZE;
		pt2 += ADD_SIZE;
		pt3 += ADD_SIZE;
		pt4 += ADD_SIZE;
		pt5 += ADD_SIZE;
		pt6 += ADD_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_add_ps(_t1, _o1);
		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		other.matrix[i] += matrix[i];
	}
	return other;
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator+(const CMtx& other)&&
{

	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;

	int i = 0;
	for (; i < elements - ADD_SIZE; i += ADD_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_add_ps(_t1, _o1);
		_t2 = _mm256_add_ps(_t2, _o2);
		_t3 = _mm256_add_ps(_t3, _o3);
		_t4 = _mm256_add_ps(_t4, _o4);
		_t5 = _mm256_add_ps(_t5, _o5);
		_t6 = _mm256_add_ps(_t6, _o6);

		_mm256_storeu_ps(&matrix[pt1], _t1);
		_mm256_storeu_ps(&matrix[pt2], _t2);
		_mm256_storeu_ps(&matrix[pt3], _t3);
		_mm256_storeu_ps(&matrix[pt4], _t4);
		_mm256_storeu_ps(&matrix[pt5], _t5);
		_mm256_storeu_ps(&matrix[pt6], _t6);

		pt1 += ADD_SIZE;
		pt2 += ADD_SIZE;
		pt3 += ADD_SIZE;
		pt4 += ADD_SIZE;
		pt5 += ADD_SIZE;
		pt6 += ADD_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_add_ps(_t1, _o1);
		_mm256_storeu_ps(&matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		matrix[i] += other.matrix[i];
	}
	return *this;
}

// MATRIX SUBTRACTION

MyAlgebra::CMtx MyAlgebra::CMtx::operator-(const CMtx& other) const&
{
	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	float* new_matrix = new float[elements];

	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;


	int i = 0;
	for (; i < elements - SUB_SIZE; i += SUB_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_sub_ps(_t1, _o1);
		_t2 = _mm256_sub_ps(_t2, _o2);
		_t3 = _mm256_sub_ps(_t3, _o3);
		_t4 = _mm256_sub_ps(_t4, _o4);
		_t5 = _mm256_sub_ps(_t5, _o5);
		_t6 = _mm256_sub_ps(_t6, _o6);

		_mm256_storeu_ps(&new_matrix[pt1], _t1);
		_mm256_storeu_ps(&new_matrix[pt2], _t2);
		_mm256_storeu_ps(&new_matrix[pt3], _t3);
		_mm256_storeu_ps(&new_matrix[pt4], _t4);
		_mm256_storeu_ps(&new_matrix[pt5], _t5);
		_mm256_storeu_ps(&new_matrix[pt6], _t6);

		pt1 += SUB_SIZE;
		pt2 += SUB_SIZE;
		pt3 += SUB_SIZE;
		pt4 += SUB_SIZE;
		pt5 += SUB_SIZE;
		pt6 += SUB_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_sub_ps(_t1, _o1);
		_mm256_storeu_ps(&new_matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		new_matrix[i] = matrix[i] - other.matrix[i];
	}
	return CMtx(rows, cols, new_matrix);
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator-(CMtx&& other) const&
{
	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;


	int i = 0;
	for (; i < elements - SUB_SIZE; i += SUB_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_sub_ps(_t1, _o1);
		_t2 = _mm256_sub_ps(_t2, _o2);
		_t3 = _mm256_sub_ps(_t3, _o3);
		_t4 = _mm256_sub_ps(_t4, _o4);
		_t5 = _mm256_sub_ps(_t5, _o5);
		_t6 = _mm256_sub_ps(_t6, _o6);

		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		_mm256_storeu_ps(&other.matrix[pt2], _t2);
		_mm256_storeu_ps(&other.matrix[pt3], _t3);
		_mm256_storeu_ps(&other.matrix[pt4], _t4);
		_mm256_storeu_ps(&other.matrix[pt5], _t5);
		_mm256_storeu_ps(&other.matrix[pt6], _t6);

		pt1 += SUB_SIZE;
		pt2 += SUB_SIZE;
		pt3 += SUB_SIZE;
		pt4 += SUB_SIZE;
		pt5 += SUB_SIZE;
		pt6 += SUB_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_sub_ps(_t1, _o1);
		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		other.matrix[i] -= matrix[i];
	}
	return other;
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator-(CMtx&& other)&&
{

	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;


	int i = 0;
	for (; i < elements - SUB_SIZE; i += SUB_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_sub_ps(_t1, _o1);
		_t2 = _mm256_sub_ps(_t2, _o2);
		_t3 = _mm256_sub_ps(_t3, _o3);
		_t4 = _mm256_sub_ps(_t4, _o4);
		_t5 = _mm256_sub_ps(_t5, _o5);
		_t6 = _mm256_sub_ps(_t6, _o6);

		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		_mm256_storeu_ps(&other.matrix[pt2], _t2);
		_mm256_storeu_ps(&other.matrix[pt3], _t3);
		_mm256_storeu_ps(&other.matrix[pt4], _t4);
		_mm256_storeu_ps(&other.matrix[pt5], _t5);
		_mm256_storeu_ps(&other.matrix[pt6], _t6);

		pt1 += SUB_SIZE;
		pt2 += SUB_SIZE;
		pt3 += SUB_SIZE;
		pt4 += SUB_SIZE;
		pt5 += SUB_SIZE;
		pt6 += SUB_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_sub_ps(_t1, _o1);
		_mm256_storeu_ps(&other.matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		other.matrix[i] -= matrix[i];
	}
	return other;
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator-(const CMtx& other)&&
{

	if (rows != other.rows || cols != other.cols)
	{
		throw std::exception("Invalid dimensions!");
	}
	// KEEP PIPELINE
	// this
	__m256 _t1, _t2, _t3, _t4, _t5, _t6;
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;

	int i = 0;
	for (; i < elements - SUB_SIZE; i += SUB_SIZE)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_t2 = _mm256_loadu_ps(&matrix[pt2]);
		_t3 = _mm256_loadu_ps(&matrix[pt3]);
		_t4 = _mm256_loadu_ps(&matrix[pt4]);
		_t5 = _mm256_loadu_ps(&matrix[pt5]);
		_t6 = _mm256_loadu_ps(&matrix[pt6]);

		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&other.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&other.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&other.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&other.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&other.matrix[pt6]);

		_t1 = _mm256_sub_ps(_t1, _o1);
		_t2 = _mm256_sub_ps(_t2, _o2);
		_t3 = _mm256_sub_ps(_t3, _o3);
		_t4 = _mm256_sub_ps(_t4, _o4);
		_t5 = _mm256_sub_ps(_t5, _o5);
		_t6 = _mm256_sub_ps(_t6, _o6);

		_mm256_storeu_ps(&matrix[pt1], _t1);
		_mm256_storeu_ps(&matrix[pt2], _t2);
		_mm256_storeu_ps(&matrix[pt3], _t3);
		_mm256_storeu_ps(&matrix[pt4], _t4);
		_mm256_storeu_ps(&matrix[pt5], _t5);
		_mm256_storeu_ps(&matrix[pt6], _t6);

		pt1 += SUB_SIZE;
		pt2 += SUB_SIZE;
		pt3 += SUB_SIZE;
		pt4 += SUB_SIZE;
		pt5 += SUB_SIZE;
		pt6 += SUB_SIZE;
	}
	for (; i < elements - VECTOR_LENGTH; i += VECTOR_LENGTH)
	{
		_t1 = _mm256_loadu_ps(&matrix[pt1]);
		_o1 = _mm256_loadu_ps(&other.matrix[pt1]);
		_t1 = _mm256_sub_ps(_t1, _o1);
		_mm256_storeu_ps(&matrix[pt1], _t1);
		pt1 += VECTOR_LENGTH;
	}
	for (; i < elements; i++)
	{
		matrix[i] -= other.matrix[i];
	}
	return *this;
}

// CHANGE SIGN OF MATRIX

MyAlgebra::CMtx MyAlgebra::CMtx::operator-() const&
{
	float* new_matrix = new float[elements];
	for (int i = 0; i < elements; i++)
	{
		new_matrix[i] = -matrix[i];
	}
	return CMtx(rows, cols, new_matrix);
}

// MATRIX TRANSPOSITION

MyAlgebra::CMtx MyAlgebra::CMtx::operator~() const&
{
	float* new_matrix = new float[elements];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			new_matrix[j * rows + i] = matrix[i * cols + j];
		}
	}
	return CMtx(cols, rows, new_matrix);
}

// MATRIX POWER

MyAlgebra::CMtx MyAlgebra::CMtx::operator^(int power) const&
{
	if (rows != cols)
	{
		throw std::exception("Rows must match columns to perform power");
	}
	else if (power < 0)
	{
		throw std::exception("Exponent can't be lower than 0");
	}
	else if (power == 0)
	{
		return CMtx(rows, 1.0f);
	}
	else if (power == 1)
	{
		return CMtx(*this);
	}

	CMtx mtx = CMtx(*this);
	for (int i = 0; i < power - 1; i++)
	{
		mtx = std::move(mtx * *this);
	}
	return mtx;
}

// MATRIX COMPARISON

bool MyAlgebra::CMtx::operator==(const CMtx& other) const&
{	
	if (rows != other.rows || cols != other.cols)
	{
		return false;
	}
	if (&other == this)
	{
		return true;
	}
	for (int i = 0; i < elements; i++)
	{
		if (abs(matrix[i] - other.matrix[i]) >= ALG_PRECISION)
		{
			return false;
		}
	}
	return true;
}

// DISPLAY MATRIX

void MyAlgebra::CMtx::display() const
{
	for (int i = 0; i < rows; i++)
	{
		std::cout << "| ";
		for (int j = 0; j < cols; j++)
		{
			std::cout << matrix[i * cols + j] << ", ";
		}
		std::cout << '|' << std::endl;
	}
}

// MULTIPLY BY SCALAR

MyAlgebra::CMtx MyAlgebra::CMtx::scalar_multiplication(const CMtx& mtx, float scalar) const&
{
	// KEEP PIPELINE
	float* new_matrix = new float[elements];
	// scalar
	__m256 _t1 = _mm256_set1_ps(scalar);
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;

	int i = 0;
	for (; i < elements - SCALAR_MULT_SIZE; i += SCALAR_MULT_SIZE)
	{
		_o1 = _mm256_loadu_ps(&mtx.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&mtx.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&mtx.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&mtx.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&mtx.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&mtx.matrix[pt6]);

		_o1 = _mm256_mul_ps(_t1, _o1);
		_o2 = _mm256_mul_ps(_t1, _o2);
		_o3 = _mm256_mul_ps(_t1, _o3);
		_o4 = _mm256_mul_ps(_t1, _o4);
		_o5 = _mm256_mul_ps(_t1, _o5);
		_o6 = _mm256_mul_ps(_t1, _o6);

		_mm256_storeu_ps(&new_matrix[pt1], _o1);
		_mm256_storeu_ps(&new_matrix[pt2], _o2);
		_mm256_storeu_ps(&new_matrix[pt3], _o3);
		_mm256_storeu_ps(&new_matrix[pt4], _o4);
		_mm256_storeu_ps(&new_matrix[pt5], _o5);
		_mm256_storeu_ps(&new_matrix[pt6], _o6);

		pt1 += SCALAR_MULT_SIZE;
		pt2 += SCALAR_MULT_SIZE;
		pt3 += SCALAR_MULT_SIZE;
		pt4 += SCALAR_MULT_SIZE;
		pt5 += SCALAR_MULT_SIZE;
		pt6 += SCALAR_MULT_SIZE;
	}
	for (; i < elements; i++)
	{
		new_matrix[i] = mtx.matrix[i] * scalar;
	}
	return CMtx(rows, cols, new_matrix);
}

MyAlgebra::CMtx MyAlgebra::CMtx::scalar_multiplication(CMtx&& mtx, float scalar)
{
	// KEEP PIPELINE
	// scalar
	__m256 _t1 = _mm256_set1_ps(scalar);
	// other
	__m256 _o1, _o2, _o3, _o4, _o5, _o6;
	// indexes
	int pt1 = 0;
	int pt2 = VECTOR_LENGTH;
	int pt3 = pt2 + VECTOR_LENGTH;
	int pt4 = pt3 + VECTOR_LENGTH;
	int pt5 = pt4 + VECTOR_LENGTH;
	int pt6 = pt5 + VECTOR_LENGTH;

	int i = 0;
	for (; i < elements - SCALAR_MULT_SIZE; i += SCALAR_MULT_SIZE)
	{
		_o1 = _mm256_loadu_ps(&mtx.matrix[pt1]);
		_o2 = _mm256_loadu_ps(&mtx.matrix[pt2]);
		_o3 = _mm256_loadu_ps(&mtx.matrix[pt3]);
		_o4 = _mm256_loadu_ps(&mtx.matrix[pt4]);
		_o5 = _mm256_loadu_ps(&mtx.matrix[pt5]);
		_o6 = _mm256_loadu_ps(&mtx.matrix[pt6]);

		_o1 = _mm256_mul_ps(_t1, _o1);
		_o2 = _mm256_mul_ps(_t1, _o2);
		_o3 = _mm256_mul_ps(_t1, _o3);
		_o4 = _mm256_mul_ps(_t1, _o4);
		_o5 = _mm256_mul_ps(_t1, _o5);
		_o6 = _mm256_mul_ps(_t1, _o6);

		_mm256_storeu_ps(&mtx.matrix[pt1], _o1);
		_mm256_storeu_ps(&mtx.matrix[pt2], _o2);
		_mm256_storeu_ps(&mtx.matrix[pt3], _o3);
		_mm256_storeu_ps(&mtx.matrix[pt4], _o4);
		_mm256_storeu_ps(&mtx.matrix[pt5], _o5);
		_mm256_storeu_ps(&mtx.matrix[pt6], _o6);

		pt1 += SCALAR_MULT_SIZE;
		pt2 += SCALAR_MULT_SIZE;
		pt3 += SCALAR_MULT_SIZE;
		pt4 += SCALAR_MULT_SIZE;
		pt5 += SCALAR_MULT_SIZE;
		pt6 += SCALAR_MULT_SIZE;
	}
	for (; i < elements; i++)
	{
		mtx.matrix[i] *= scalar;
	}
	return mtx;
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator*(float scalar) const&
{
	return scalar_multiplication(*this, scalar);
}

MyAlgebra::CMtx MyAlgebra::CMtx::operator*(float scalar)&&
{
	return scalar_multiplication(*this, scalar);
}

MyAlgebra::CMtx MyAlgebra::operator*(float scalar, const CMtx& other)
{
	return other * scalar;
}

MyAlgebra::CMtx MyAlgebra::operator*(float scalar, CMtx&& other)
{
	return other * scalar;
}

// NAME

std::string MyAlgebra::CMtx::authorName()
{
	return "Konrad_Karanowski";
}
