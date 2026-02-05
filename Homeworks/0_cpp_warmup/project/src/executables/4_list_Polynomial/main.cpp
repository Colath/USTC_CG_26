#include "PolynomialList.h"

#include <list>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
	PolynomialList p1("D:/USTC_CG_26/Homeworks/0_cpp_warmup/project/data/P3.txt");
	PolynomialList p2("D:/USTC_CG_26/Homeworks/0_cpp_warmup/project/data/P4.txt");
	PolynomialList p3;
	p1.Print();
	p2.Print();

	p3 = p1 + p2;
	p3.Print();
	p3 = p1 - p2;
	p3.Print();

	p3 = p1 * p2;
	p3.Print();

	return 0;
}