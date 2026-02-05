// implementation of class DArray
#include "DArray.h"
#include<iostream>
#include<cassert>

// default constructor
DArray::DArray() {
	Init();
}

// set an array with default values
DArray::DArray(int nSize, double dValue) {
	m_nSize = nSize;
	m_nMax = nSize;
	m_pData = new double[nSize];
	for (int i = 0; i < m_nSize; i++) {
		m_pData[i] = dValue;
	}
}

DArray::DArray(const DArray& arr)
	:m_nSize(arr.m_nSize), m_nMax(arr.m_nMax), m_pData(new double[arr.m_nMax]) {
	for (int i = 0; i < m_nSize; i++)
		m_pData[i] = arr.m_pData[i];
}

// deconstructor
DArray::~DArray() {
	Free();
}

// display the elements of the array
void DArray::Print() const {
	for (int i = 0; i < m_nSize; i++)
		std::cout << m_pData[i] << " ";
	std::cout << std::endl;
}

// initilize the array
void DArray::Init() {
	m_pData = nullptr;
	m_nSize = 0;
	m_nMax = 0;
}

// free the array
void DArray::Free() {
	delete[] m_pData;
	m_pData = nullptr;
	m_nMax = 0;
	m_nSize = 0;
}

// get the size of the array
int DArray::GetSize() const {
	return m_nSize;
}

// set the size of the array
void DArray::SetSize(int nSize) {
	if (nSize < 0)
		return;
	else if (nSize > m_nMax) {
		int newcap = m_nMax == 0 ? 1 : m_nMax;
		while (newcap < nSize)
			newcap *= 2;
		double* newdata = new double[newcap];
		for (int i = 0; i < m_nSize; i++)
			newdata[i] = m_pData[i];
		for (int i = m_nSize; i < nSize; i++)
			newdata[i] = 0.0;
		delete[] m_pData;
		m_pData = newdata;
		m_nMax = newcap;
	}
	else if (nSize > m_nSize) {
		for (int i = m_nSize; i < nSize; i++)
			m_pData[i] = 0.0;
	}
	m_nSize = nSize;
}

void DArray::Reserve(int nSize) {
	if (nSize <= m_nMax)
		return;

	double* newData = new double[nSize];
	for (int i = 0; i < m_nSize; i++) {
		newData[i] = m_pData[i];
	}

	delete[] m_pData;
	m_pData = newData;
	m_nMax = nSize;
}


// get an element at an index
const double& DArray::GetAt(int nIndex) const {
	assert(nIndex >= 0 && nIndex < m_nSize);
	return m_pData[nIndex];
}

// set the value of an element 
void DArray::SetAt(int nIndex, double dValue) {
	if (nIndex < 0 || nIndex >= m_nSize) {
		return;
	}
	m_pData[nIndex] = dValue;
}

// overload operator '[]'
double& DArray::operator[](int nIndex) {
	assert(nIndex >= 0 && nIndex < m_nSize);
	return m_pData[nIndex];
}

// overload operator '[]'
const double& DArray::operator[](int nIndex) const {
	assert(nIndex >= 0 && nIndex < m_nSize);
	return m_pData[nIndex];
}

// add a new element at the end of the array
void DArray::PushBack(double dValue) {
	if (m_nSize == m_nMax) {
		if (m_nMax == 0)
			m_nMax = 1;
		else
			m_nMax *= 2;
		double* newData = new double[m_nMax];
		for (int i = 0; i < m_nSize; i++) {
			newData[i] = m_pData[i];
		}
		newData[m_nSize] = dValue;
		delete[] m_pData;
		m_pData = newData;
		m_nSize++;
	}
	else {
		m_pData[m_nSize] = dValue;
		m_nSize++;
	}
}

// delete an element at some index
void DArray::DeleteAt(int nIndex) {
	assert(nIndex >= 0 && nIndex < m_nSize);
	for (int i = nIndex; i < m_nSize - 1; i++) {
		m_pData[i] = m_pData[i + 1];
	}
	m_nSize--;
}

// insert a new element at some index
void DArray::InsertAt(int nIndex, double dValue) {
	assert(nIndex >= 0 && nIndex <= m_nSize);
	if (m_nSize == m_nMax) {
		if (m_nMax == 0)
			m_nMax = 1;
		else
			m_nMax *= 2;
		double* newData = new double[m_nMax];
		for (int i = 0; i < nIndex; i++) {
			newData[i] = m_pData[i];
		}
		newData[nIndex] = dValue;
		for (int i = nIndex; i < m_nSize; i++) {
			newData[i + 1] = m_pData[i];
		}
		delete[] m_pData;
		m_pData = newData;
		m_nSize++;
	}
	else {
		for (int i = m_nSize; i > nIndex; i--) {
			m_pData[i] = m_pData[i - 1];
		}
		m_pData[nIndex] = dValue;
		m_nSize++;
	}
}

// overload operator '='
DArray& DArray::operator=(const DArray& arr) {
	if (this == &arr) return *this;

	double* newData = (arr.m_nMax > 0) ? new double[arr.m_nMax] : nullptr;
	for (int i = 0; i < arr.m_nSize; ++i)
		newData[i] = arr.m_pData[i];

	delete[] m_pData;
	m_pData = newData;
	m_nSize = arr.m_nSize;
	m_nMax = arr.m_nMax;

	return *this;
}
