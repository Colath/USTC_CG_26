#include "PolynomialList.h"
#include<cmath>
#include<assert.h>
#include<fstream>
#include<iostream>
#include<algorithm>

#define EPSILON 1.0e-6

using namespace std;

PolynomialList::PolynomialList(const PolynomialList& other) {
    m_Polynomial = other.m_Polynomial;
}

PolynomialList::PolynomialList(const string& file) {
    ReadFromFile(file);
}

PolynomialList::PolynomialList(const double* cof, const int* deg, int n) {
    for (int i = 0; i < n; i++) {
        AddOneTerm(Term(deg[i], cof[i]));
    }
}

PolynomialList::PolynomialList(const vector<int>& deg, const vector<double>& cof) {
    assert(deg.size() == cof.size());
    for (int i = 0; i < deg.size(); i++) {
        AddOneTerm(Term(deg[i], cof[i]));
    }
}

double PolynomialList::coff(int i) const {
    for (const Term& term : m_Polynomial) {
        if (term.deg == i)
            return term.cof;
    }
    return 0.0;
}

double& PolynomialList::coff(int i) {
    return AddOneTerm(Term(i, 0)).cof;
}

void PolynomialList::compress() {
    m_Polynomial.sort([](const Term& a, const Term& b) {
        return a.deg > b.deg;
        });
    if (m_Polynomial.empty()) return;
    auto it = m_Polynomial.begin();
    auto next_it = it;
    next_it++;

    while (next_it != m_Polynomial.end()) {
        if (it->deg == next_it->deg) {
            it->cof += next_it->cof;
            next_it = m_Polynomial.erase(next_it);
        }
        else {
            it++;
            next_it = it;
            next_it++;
        }
    }
    auto it_0 = m_Polynomial.begin();
    while (it_0 != m_Polynomial.end()) {
        if (fabs((*it_0).cof) < EPSILON)
            it_0 = m_Polynomial.erase(it_0);
        else
            it_0++;
    }
}

PolynomialList PolynomialList::operator+(const PolynomialList& right) const {
    PolynomialList poly(*this);
    for (const auto& term : right.m_Polynomial)
        poly.AddOneTerm(term);
    poly.compress();
    return poly;
}

PolynomialList PolynomialList::operator-(const PolynomialList& right) const {
    PolynomialList poly = *this;
    for (const auto& term : right.m_Polynomial)
        poly.AddOneTerm(Term(term.deg, -term.cof));
    poly.compress();
    return poly;
}

PolynomialList PolynomialList::operator*(const PolynomialList& right) const {
    PolynomialList poly;
    for (const auto& term_1 : m_Polynomial) {
        for (const auto& term_2 : right.m_Polynomial) {
            poly.AddOneTerm(Term(term_1.deg + term_2.deg, term_1.cof * term_2.cof));
        }
    }
    poly.compress();
    return poly;
}

PolynomialList& PolynomialList::operator=(const PolynomialList& right) {
    m_Polynomial = right.m_Polynomial;
    return *this;
}

void PolynomialList::Print() const {
    auto it = m_Polynomial.begin();

    while (it != m_Polynomial.end()) {
        if (it == m_Polynomial.begin()) {
            if (it->cof < 0) cout << "-";
        }
        else {
            if (it->cof >= 0) cout << " + ";
            else              cout << " - ";
        }
        cout << " " << fabs(it->cof);
        if (it->deg > 0) cout << "x";
        if (it->deg > 1) cout << "^" << it->deg;
        cout << " ";
        it++;
    }
    cout << endl;
}

bool PolynomialList::ReadFromFile(const string& file) {
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
        Term nd;
        inp >> nd.deg;
        inp >> nd.cof;

        AddOneTerm(nd);
    }

    inp.close();

    return true;
}

PolynomialList::Term& PolynomialList::AddOneTerm(const Term& term) {
    for (auto it = m_Polynomial.begin(); it != m_Polynomial.end(); ++it) {
        if (it->deg == term.deg) {
            it->cof += term.cof;
            return *it;
        }
        if (it->deg < term.deg) {
            auto newIt = m_Polynomial.insert(it, term);
            return *newIt;
        }
    }
    m_Polynomial.push_back(term);
    return m_Polynomial.back();
}