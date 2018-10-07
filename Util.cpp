#include "Util.h"
bool comparator(const intDoublePair& l, const intDoublePair& r)
{
	return l.second < r.second;
}
void rotate2D(double &vx, double &vy, double theta) {
	double cosa = cos(theta), sina = sin(theta);
	//rotate goal heading by angle to get this heading
	double vxtemp = vx*cosa + vy*-sina; 
	double vytemp = vx*sina + vy*cosa;
	vx = vxtemp;
	vy = vytemp;
}
std::vector<intDoublePair> sortWithIndexReturn(std::vector<double> toSort) {
	std::vector<intDoublePair> sorted(toSort.size());
	for (int i = 0; i < toSort.size(); i++) {
		sorted[i] = intDoublePair(i, toSort[i]);
	}
	std::sort(sorted.begin(),sorted.end(),comparator);
	return sorted;
}
std::vector<intDoublePair> sortWithIndexReturn(std::list<double> toSort) {
	std::vector<intDoublePair> sorted(toSort.size());
	int ix = 0;
	for (std::list<double>::iterator iter = toSort.begin(); iter != toSort.end(); ++iter) {
		sorted[ix] = intDoublePair(ix, (*iter));
		ix++;
	}	
	std::sort(sorted.begin(), sorted.end(),comparator);
	return sorted;
}
void softmax(std::vector<double> &x){
	double esum = 0;
	for (int i = 0; i < x.size(); i++) {
		x[i] = std::exp(x[i]);
		esum += x[i];
	}
	for (int i = 0; i < x.size(); i++) {
		x[i] = x[i] / esum;
	}
}
int argmin(std::vector<double> &x) {
	int amin = -1;
	double min = 9e99;
	for (int i = 0; i < x.size(); i++) {
		if (x[i] < min) {
			amin = i;
			min = x[i];
		}
	}
	return amin;
}
void normalize(double &vx, double &vy) {
	double norm = std::sqrt(vx*vx + vy*vy);
	vy = vy / norm;
	vx = vx / norm;
}
std::vector<std::string> split_string(std::string s, char delim) {
	std::istringstream ss(s);

	std::vector<std::string> parts;
	std::string part;
	while (std::getline(ss, part, delim)) {
		if (!part.empty()) {
			parts.push_back(part);
		}
	}

	return parts;
}
