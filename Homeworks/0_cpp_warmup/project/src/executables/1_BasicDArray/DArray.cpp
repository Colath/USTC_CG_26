// implementation of class DArray
#include "DArray.h"
#include <iostream>

// default constructor
DArray::DArray() {
	Init();
}

// set an array with default values
DArray::DArray(int nSize, double dValue) {
	m_nSize = nSize;
	m_pData = new double[nSize];
	for (int i = 0; i < nSize; i++) {
		m_pData[i] = dValue;
	}
}

DArray::DArray(const DArray& arr) {
	m_nSize = arr.m_nSize;
	m_pData = new double[m_nSize];
	for (int i = 0; i < m_nSize; i++) {
		m_pData[i] = arr.m_pData[i];
	}
}

// deconstructor
DArray::~DArray() {
	Free();
}

// display the elements of the array
void DArray::Print() const {
	if (m_nSize == 0) {
		std::cout << "Empty Array!" << std::endl;
	}
	else if (m_pData == nullptr) {
		std::cout << "Error" << std::endl;
	}
	else {
		for (int i = 0; i < m_nSize; i++) {
			std::cout << m_pData[i] << " ";
		}
		std::cout << std::endl;
	}
}

// initilize the array
void DArray::Init() {
	m_nSize = 0;
	m_pData = nullptr;
}

// free the array
void DArray::Free() {
	if (m_pData != nullptr) {
		delete[] m_pData;
		m_pData = nullptr;
		m_nSize = 0;
	}
}

// get the size of the array
int DArray::GetSize() const {
	return m_nSize; // you should return a correct value
}

// set the size of the array
void DArray::SetSize(int nSize) {
	if (nSize < 0) nSize = 0;
	if (nSize == m_nSize) return;

	double* newData = (nSize > 0) ? new double[nSize] : nullptr;

	int copyCount = (nSize < m_nSize) ? nSize : m_nSize;
	for (int i = 0; i < copyCount; ++i) newData[i] = m_pData[i];
	for (int i = copyCount; i < nSize; ++i) newData[i] = 0.0;

	delete[] m_pData;
	m_pData = newData;
	m_nSize = nSize;
}


// get an element at an index
const double& DArray::GetAt(int nIndex) const {
	if (nIndex < 0 || nIndex >= m_nSize) {
		std::cout << "Error" << std::endl;
		static double dummy = 0.0;
		return dummy;
	}
	else {
		return m_pData[nIndex];
	}
}

// set the value of an element 
void DArray::SetAt(int nIndex, double dValue) {
	if (nIndex > (m_nSize - 1) || nIndex < 0) {
		std::cout << "Error" << std::endl;
	}
	else {
		m_pData[nIndex] = dValue;
	}
}

// overload operator '[]'
const double& DArray::operator[](int nIndex) const {
	if (nIndex > (m_nSize - 1) || nIndex < 0) {
		static double dummy = 0.0;
		return dummy;
	}
	else {
		return m_pData[nIndex];
	}
}

// add a new element at the end of the array
void DArray::PushBack(double dValue) {
	double* newData = new double[m_nSize + 1];
	for (int i = 0; i < m_nSize; i++) {
		newData[i] = m_pData[i];
	}
	newData[m_nSize] = dValue;
	delete[] m_pData;
	m_pData = newData;
	++m_nSize;
}

// delete an element at some index
void DArray::DeleteAt(int nIndex) {
	if (m_nSize == 0) return;
	if (nIndex < 0 || nIndex >= m_nSize) return;
	if (m_nSize == 1) {
		delete[] m_pData;
		m_pData = nullptr;
		m_nSize = 0;
		return;
	}
	double* newData = new double[m_nSize - 1];
	for (int i = 0; i < nIndex; i++) {
		newData[i] = m_pData[i];
	}
	for (int i = nIndex + 1; i < m_nSize; i++) {
		newData[i - 1] = m_pData[i];
	}
	delete[] m_pData;
	m_pData = newData;
	m_nSize--;
}

// insert a new element at some index
void DArray::InsertAt(int nIndex, double dValue) {
	if (nIndex < 0) nIndex = 0;
	if (nIndex > m_nSize) nIndex = m_nSize;
	double* newData = new double[m_nSize + 1];
	for (int i = 0; i < nIndex; i++) {
		newData[i] = m_pData[i];
	}
	newData[nIndex] = dValue;
	for (int i = nIndex; i < m_nSize; i++) {
		newData[i + 1] = m_pData[i];
	}
	delete[] m_pData;
	m_pData = newData;
	++m_nSize;
}

// overload operator '='
DArray& DArray::operator = (const DArray& arr) {
	if (this == &arr) {
		return *this;
	}
	delete[] m_pData;
	m_nSize = arr.m_nSize;
	m_pData = new double[m_nSize];
	for (int i = 0; i < m_nSize; i++) {
		m_pData[i] = arr.m_pData[i];
	}
	return *this;
}
