#pragma once
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdio.h>

template <class T>
T getAt(std::list<T> source, int ix) {
	T toReturn;
	int i = 0;
	typename std::list<T>::iterator iter = source.begin();
	while (iter != source.end() && i < ix) {
		i++;
		++iter;
	}
	toReturn = (*iter);
	return toReturn;
};

typedef std::pair<int, double> intDoublePair;
bool comparator(const intDoublePair& l, const intDoublePair& r);
std::vector<intDoublePair> sortWithIndexReturn(std::vector<double> toSort);
std::vector<intDoublePair> sortWithIndexReturn(std::list<double> toSort);
void softmax(std::vector<double> &x);
void rotate2D(double &vx, double &vy, double theta);
void normalize(double &vx, double &vy);
int argmin(std::vector<double> &x);
std::vector<std::string> split_string(std::string s, char delim);