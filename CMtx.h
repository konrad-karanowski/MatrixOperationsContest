#pragma once
#include <string>

namespace MyAlgebra
{
	class CMtx
	{
	private:
		int     rows;
		int     cols;
		int	 elements;
		float* matrix;

		// helper constructor
		CMtx(uint16_t rows, uint16_t cols, float* matrix);

		// =========================================================================
		// MATRIX MULTIPLICATION:
		// =========================================================================
		
		// matmul
		void matmul(int idx0, int idx1, const CMtx& this_transposed, const CMtx& other, float* result) const;

		// multiplication by scalar
		CMtx scalar_multiplication(const CMtx& mtx, float scalar) const&;
		CMtx scalar_multiplication(CMtx&& mtx, float scalar);

	public:
		// global parameters
		static const float ALG_PRECISION;
		static const int N_JOBS;
		static const int VECTOR_LENGTH;
		// SIMD parameters
		static const int ADD_SIZE;
		static const int SUB_SIZE;
		static const int MULT_SIZE;
		static const int SCALAR_MULT_SIZE;
		// thresholds
		static const int THRESHOLD_MUL;

		// =========================================================================
		// KONSTRUKTORY:
		// =========================================================================

		// Tworzy macierz z możliwością losowej inicjalizacji
		CMtx(uint16_t rows, uint16_t cols, bool rand_init = false);

		// Tworzy kwadratową macierz diagonalną
		CMtx(uint16_t rows, float diagonal);

		// konstruktor kopiujący
		CMtx(const CMtx& other);

		// konstruktor przenoszący
		CMtx(CMtx&& other);

		// Jeśli potrzeba - należy zadeklarować i zaimplementować inne konstruktory
		~CMtx();

		// =========================================================================
		// OPERATORY PRZYPISANIA:
		// =========================================================================

		const CMtx& operator=(const CMtx& other);

		// Zamiana macierzy na macierz diagonalną 
		const CMtx& operator=(float diagonal);

		// Operator przenoszący
		const CMtx& operator=(CMtx&& other);


		// =========================================================================
		// INDEKSOWANIE MACIERZY
		// =========================================================================

		float* operator[](int idx);

		// =========================================================================
		// OPERACJE ALGEBRAICZNE
		// =========================================================================

		// mnożenie macierzy
		CMtx operator*(const CMtx& other) const&;

		// Mnożenie macierzy przez stałą
		CMtx operator*(float scalar) const&;
		CMtx operator*(float scalar) &&;

		// Dodawanie macierzy
		CMtx operator+(const CMtx& other) const&;
		CMtx operator+(const CMtx& other)&&;
		CMtx operator+(CMtx&& other) const&;
		CMtx operator+(CMtx&& other)&&;

		// Odejmowanie macierzy
		CMtx operator-(const CMtx& other) const&;
		CMtx operator-(const CMtx& other) &&;
		CMtx operator-(CMtx&& other) const&;
		CMtx operator-(CMtx&& other) &&;

		// Minus unarny - zmiana znaku wszystkich współczynników macierzy
		CMtx operator-() const&;

		// Transponowanie macierzy
		CMtx operator~() const&;

		// Akceptuje tylko power >= 0:
		//    power = 0  - zwraca macierz jednostkową
		//    power = 1  - zwraca kopię macierzy
		//    power > 1  - zwraca iloczyn macierzy 
		CMtx operator^(int power) const&;

		// Porównywanie macierzy z dokładnością do stałej ALG_PRECISION
		bool operator==(const CMtx& other) const&;

		// Tylko do celów testowych - wypisuje macierz wierszami na stdout
		void display() const;

		std::string authorName();
	};

	CMtx operator*(float multiplier, const CMtx& other);
	CMtx operator*(float multiplier, CMtx&& other);
}


