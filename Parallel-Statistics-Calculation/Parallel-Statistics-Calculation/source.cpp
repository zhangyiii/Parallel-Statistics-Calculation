#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Utils.h"
#include <future>
#include <CL/cl.hpp>

//Function prototypes
int findIndex(int value, vector<int> &temp);

int main()
{//Program entry point
	ifstream dataFile;
	string data, filename;
	int count = 0;
	int platform_id = 0;
	int device_id = 0;

	//Result variables
	double mean;
	int min = 0, max = 0;
	future<int> fumax, fumin, fumean;
	int index_min = 0, index_max = 0, index_mean = 0;

	//Vectors to hold data from all weather stations
	vector<string>stationName;
	vector<int>year;
	vector<int>month;
	vector<int>day;
	vector<int>time;
	vector<int>temperature;

	std::cout << "Enter relative filepath: ";
	std::cin >> filename;

	//Open file and check for failure file
	dataFile.open(filename, std::ofstream::out | std::ofstream::app);
	if (!dataFile.is_open()) {
		cout << "File io error... Program will now close." << endl;
		system("pause");
		exit(0);
	}

	//Read in all data
	std::cout << "Reading in data..." << std::endl;
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
			temperature.push_back(stoi(data));
			count = 0;
			break;
		}

	} dataFile.close();
	
	/* Data is now all read in and stored into vectors. Calculations can now be performed */

	//Select / Output computing devices and initiate command queue 
	cl::Context context = GetContext(platform_id, device_id);
	std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
	cl::CommandQueue queue(context);

	//Load & build the device code
	cl::Program::Sources sources;
	AddSources(sources, "kernel.cl");
	cl::Program program(context, sources);

	//Build kernal code, catch any errors
	try {
		program.build();
	}
	catch (const cl::Error& err) {
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
		throw err;
	}

	size_t vector_elements = temperature.size();
	size_t vector_size = temperature.size()*sizeof(int);

    //Output vector
	std::vector<int> result_min(vector_elements);
	std::vector<int> result_max(vector_elements);
	std::vector<int> result_mean(vector_elements);

	//Device buffers
	cl::Buffer buffer_temperature(context, CL_MEM_READ_WRITE, vector_size);
	cl::Buffer buffer_result(context, CL_MEM_READ_WRITE, vector_size);

	/* - Max -*/

	//Copy arrays to device memory
	queue.enqueueWriteBuffer(buffer_temperature, CL_TRUE, 0, vector_size, &temperature[0]);
	queue.enqueueWriteBuffer(buffer_result, CL_TRUE, 0, vector_size, &result_max[0]);

	//Setup and execute the kernel
	cl::Kernel kernel_max = cl::Kernel(program, "reduce_max");
	kernel_max.setArg(0, buffer_temperature);
	kernel_max.setArg(1, buffer_result);
	kernel_max.setArg(2, cl::Local(5*sizeof(int)));

	queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);
	
	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, vector_size, &result_max[0]);
	max = result_max[0];

	/* - Min -*/

	//Copy arrays to device memory
	queue.enqueueWriteBuffer(buffer_temperature, CL_TRUE, 0, vector_size, &temperature[0]);
	queue.enqueueWriteBuffer(buffer_result, CL_TRUE, 0, vector_size, &result_max[0]);

	//Setup and execute the kernel
	cl::Kernel kernel_min = cl::Kernel(program, "reduce_min");
	kernel_min.setArg(0, buffer_temperature);
	kernel_min.setArg(1, buffer_result);
	kernel_min.setArg(2, cl::Local(5 * sizeof(int)));

	queue.enqueueNDRangeKernel(kernel_min, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, vector_size, &result_min[0]);
	min = result_min[0];

	/* - Mean -*/

	//Copy arrays to device memory
	queue.enqueueWriteBuffer(buffer_temperature, CL_TRUE, 0, vector_size, &temperature[0]);
	queue.enqueueWriteBuffer(buffer_result, CL_TRUE, 0, vector_size, &result_max[0]);

	//Setup and execute the kernel
	cl::Kernel kernel_mean = cl::Kernel(program, "reduce_avg");
	kernel_mean.setArg(0, buffer_temperature);
	kernel_mean.setArg(1, buffer_result);
	kernel_mean.setArg(2, cl::Local(5 * sizeof(int)));

	queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);

	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, vector_size, &result_mean[0]);
	min = result_mean[0];









	/* Find index's */

	//Start functions
	fumax = std::async(findIndex, max, temperature);
	fumin = std::async(findIndex, min, temperature);

	//Wait for async functions to return
	index_max = fumax.get();
	index_min = fumin.get();

	//Output result
	std::cout << "\n-----------------------------------\n" << std::endl;
	std::cout << "Max: " << max << " first occured at - " << stationName.at(index_max) << " on " << day.at(index_max) << " / " << month.at(index_max) << " / " << year.at(index_max) << std::endl;
	std::cout << "Min: " << min << " first occured at - " << stationName.at(index_min) << " on " << day.at(index_min) << " / " << month.at(index_min) << " / " << year.at(index_min) << std::endl;
	//std::cout << "Mean: " << mean << std::endl;


	system("pause");
	return 0;
}

int findIndex(int value, vector<int> &temp)
{//Find index position of value in vector
	for (int i = 0; i < temp.size(); i++) {
		if (value = temp.at(i)) 
			return i;
	}
}