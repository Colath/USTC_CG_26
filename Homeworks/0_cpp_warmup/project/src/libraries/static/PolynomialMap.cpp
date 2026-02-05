#include "PolynomialMap.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>

#define EPSILON 1e-6

using namespace std;

PolynomialMap::PolynomialMap(const PolynomialMap& other) {
	m_Polynomial = other.m_Polynomial;
}

PolynomialMap::PolynomialMap(const string& file) {
	ReadFromFile(file);
}

PolynomialMap::PolynomialMap(const double* cof, const int* deg, int n) {
	for (int i = 0; i < n; i++) {
		coff(deg[i]) = cof[i];
	}
}

PolynomialMap::PolynomialMap(const vector<int>& deg, const vector<double>& cof) {
	assert(deg.size() == cof.size());
	for (size_t i = 0; i < deg.size(); i++) {
		coff(deg[i]) = cof[i];
	}
}

double PolynomialMap::coff(int i) const {
	auto target = m_Polynomial.find(i);
	if (target != m_Polynomial.end())
		return target->second;
	return 0.0;
}

double& PolynomialMap::coff(int i) {
	return m_Polynomial[i];
}

void PolynomialMap::compress() {
	auto it = m_Polynomial.begin();
	while (it != m_Polynomial.end()) {
		if (fabs(it->second) < EPSILON)
			it = m_Polynomial.erase(it);
		else
			++it;
	}
}

PolynomialMap PolynomialMap::operator+(const PolynomialMap& right) const {
	PolynomialMap poly = right;
	for (const auto& term : m_Polynomial) {
		poly.coff(term.first) += term.second;
	}
	poly.compress();
	return poly;
}

PolynomialMap PolynomialMap::operator-(const PolynomialMap& right) const {
	PolynomialMap poly = right;
	for (const auto& term : m_Polynomial) {
		poly.coff(term.first) -= term.second;
	}
	poly.compress();
	return poly;
}

PolynomialMap PolynomialMap::operator*(const PolynomialMap& right) const {
	PolynomialMap poly;
	for (const auto& it1 : m_Polynomial) {
		for (const auto& it2 : right.m_Polynomial) {
			poly.coff(it1.first + it2.first) += (it1.second) * (it2.second);
		}
	}
	poly.compress();
	return poly;
}

PolynomialMap& PolynomialMap::operator=(const PolynomialMap& right) {
	m_Polynomial = right.m_Polynomial;
	return *this;
}

void PolynomialMap::Print() const {
	auto it = m_Polynomial.begin();
	while (it != m_Polynomial.end()) {
		if (it == m_Polynomial.begin()) {
			if (it->second < 0)
				cout << "-";
		}
		else {
			if (it->second > 0)
				cout << "+";
			else
				cout << "-";
		}
		cout << " " << fabs(it->second);
		if (it->first > 0) cout << "x";
		if (it->first > 1) cout << "^" << it->first;
		cout << " ";
		++it;
	}
	cout << endl;
}

bool PolynomialMap::ReadFromFile(const string& file) {
	m_Polynomial.clear();

	ifstream inp;
	inp.open(file.c_str());
	if (!inp.is_open()) {
		cout << "ERROR::PolynomialList::ReadFromFile:" << endl
			<< "\t" << "file [" << file << "] opens failed" << endl;
		return false;
	}

	char ch;
	int n;
	inp >> ch;
	inp >> n;
	for (int i = 0; i < n; i++) {
		int deg;
		double cof;
		inp >> deg;
		inp >> cof;
		coff(deg) = cof;
	}

	inp.close();

	return true;
}
