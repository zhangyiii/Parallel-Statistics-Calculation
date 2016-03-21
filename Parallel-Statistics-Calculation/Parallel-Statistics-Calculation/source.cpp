#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

using namespace std;

int main()
{//Program entry point
	ifstream dataFile;
	string data, filename;
	int count = 0;

	//Vectors to hold data from all weather stations
	vector<string>stationName;
	vector<int>year;
	vector<int>month;
	vector<int>day;
	vector<int>time;
	vector<double>temperature;

	cout << "Enter relative filepath: ";
	cin >> filename;

	//Open file and check for failure file
	dataFile.open(filename, std::ofstream::out | std::ofstream::app);
	if (!dataFile.is_open()) {
		cout << "File io error... Program will now close." << endl;
		system("pause");
		exit(0);
	}

	//Read in all data
	cout << "Reading in data..." << endl;
	while (dataFile >> data) {

		//Append data to the relavent vector
		switch (count) {
		case 0:
			stationName.push_back(data);
			count++;
			break;
		case 1:
			year.push_back(stoi(data));
			count++;
			break;
		case 2:
			month.push_back(stoi(data));
			count++;
			break;
		case 3:
			day.push_back(stoi(data));
			count++;
			break;
		case 4:
			time.push_back(stoi(data));
			count++;
			break;
		case 5:
			temperature.push_back(stod(data));
			count = 0;
			break;
		}

	}
	dataFile.close();

	return 0;
}