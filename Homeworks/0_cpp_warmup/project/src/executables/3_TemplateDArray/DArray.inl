#include <iostream>
#include <cassert>

// default constructor
template<class T>
DArray<T>::DArray() {
    Init();
}

// set an array with default values
template<class T>
DArray<T>::DArray(int nSize, const T& dValue) {
    m_nSize = nSize;
    m_nMax = nSize;
    m_pData = (nSize > 0) ? new T[nSize] : nullptr;
    for (int i = 0; i < m_nSize; i++) {
        m_pData[i] = dValue;
    }
}

// copy constructor
template<class T>
DArray<T>::DArray(const DArray& arr)
    : m_pData((arr.m_nMax > 0) ? new T[arr.m_nMax] : nullptr),
    m_nSize(arr.m_nSize),
    m_nMax(arr.m_nMax) {
    for (int i = 0; i < m_nSize; i++) {
        m_pData[i] = arr.m_pData[i];
    }
}

// deconstructor
template<class T>
DArray<T>::~DArray() {
    Free();
}

// print the elements of the array
template<class T>
void DArray<T>::Print() const {
    for (int i = 0; i < m_nSize; i++) {
        std::cout << m_pData[i] << " ";
    }
    std::cout << std::endl;
}

// allocate enough memory (capacity >= nSize), does NOT change size
template<class T>
void DArray<T>::Reserve(int nSize) {
    if (nSize <= m_nMax) return;
    int newcap = (m_nMax == 0) ? 1 : m_nMax;
    while (newcap < nSize) newcap *= 2;

    T* newdata = new T[newcap];
    for (int i = 0; i < m_nSize; i++) {
        newdata[i] = m_pData[i];
    }

    delete[] m_pData;
    m_pData = newdata;
    m_nMax = newcap;
}

// get the size of the array
template<class T>
int DArray<T>::GetSize() const {
    return m_nSize;
}

// set the size of the array
template<class T>
void DArray<T>::SetSize(int nSize) {
    if (nSize < 0) return;

    if (nSize > m_nMax) {
        Reserve(nSize);
    }

    // value-initialize newly exposed elements
    if (nSize > m_nSize) {
        for (int i = m_nSize; i < nSize; i++) {
            m_pData[i] = static_cast<T>(0);
        }
    }

    m_nSize = nSize;
}

// get an element at an index
template<class T>
const T& DArray<T>::GetAt(int nIndex) const {
    assert(nIndex >= 0 && nIndex < m_nSize);
    return m_pData[nIndex];
}

// set the value of an element
template<class T>
void DArray<T>::SetAt(int nIndex, const T& dValue) {
    if (nIndex < 0 || nIndex >= m_nSize) return;
    m_pData[nIndex] = dValue;
}

// overload operator '[]'
template<class T>
T& DArray<T>::operator[](int nIndex) {
    assert(nIndex >= 0 && nIndex < m_nSize);
    return m_pData[nIndex];
}

// overload operator '[]' (const)
template<class T>
const T& DArray<T>::operator[](int nIndex) const {
    assert(nIndex >= 0 && nIndex < m_nSize);
    return m_pData[nIndex];
}

// add a new element at the end of the array
template<class T>
void DArray<T>::PushBack(const T& dValue) {
    if (m_nSize == m_nMax) {
        Reserve((m_nMax == 0) ? 1 : (m_nMax * 2));
    }
    m_pData[m_nSize++] = dValue;
}

// delete an element at some index
template<class T>
void DArray<T>::DeleteAt(int nIndex) {
    assert(nIndex >= 0 && nIndex < m_nSize);
    for (int i = nIndex; i < m_nSize - 1; i++) {
        m_pData[i] = m_pData[i + 1];
    }
    --m_nSize;
}

// insert a new element at some index
template<class T>
void DArray<T>::InsertAt(int nIndex, const T& dValue) {
    assert(nIndex >= 0 && nIndex <= m_nSize);

    if (m_nSize == m_nMax) {
        Reserve((m_nMax == 0) ? 1 : (m_nMax * 2));
    }

    for (int i = m_nSize; i > nIndex; --i) {
        m_pData[i] = m_pData[i - 1];
    }
    m_pData[nIndex] = dValue;
    ++m_nSize;
}

// overload operator '='
template<class T>
DArray<T>& DArray<T>::operator=(const DArray& arr) {
    if (this == &arr) return *this;

    T* newData = (arr.m_nMax > 0) ? new T[arr.m_nMax] : nullptr;
    for (int i = 0; i < arr.m_nSize; ++i) {
        newData[i] = arr.m_pData[i];
    }

    delete[] m_pData;
    m_pData = newData;
    m_nSize = arr.m_nSize;
    m_nMax = arr.m_nMax;

    return *this;
}

// initilize the array
template<class T>
void DArray<T>::Init() {
    m_pData = nullptr;
    m_nSize = 0;
    m_nMax = 0;
}

// free the array
template<class T>
void DArray<T>::Free() {
    delete[] m_pData;
    m_pData = nullptr;
    m_nMax = 0;
    m_nSize = 0;
}
