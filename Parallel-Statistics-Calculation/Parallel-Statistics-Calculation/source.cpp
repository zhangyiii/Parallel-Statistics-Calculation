#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Utils.h"
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

int main()
{//Program entry point
	ifstream dataFile;
	string data, filename;
	int count = 0;
	int platform_id = 0;
	int device_id = 0;

	//Vectors to hold data from all weather stations
	vector<string>stationName;
	vector<int>year;
	vector<int>month;
	vector<int>day;
	vector<int>time;
	vector<float>temperature;

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
			temperature.push_back(stof(data));
			count = 0;
			break;
		}

	} dataFile.close();
	
	/* Data is now all read in and stored into vectors. Calculations can now be performed */

	//Select /  Output computing devices and initiate command queue 
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

	size_t vector_elements = temperature.size();//number of elements
	size_t vector_size = temperature.size()*sizeof(float);//size in bytes

    //Output vector
	std::vector<float> result(vector_elements);

	//Device buffers
	cl::Buffer buffer_temperature(context, CL_MEM_READ_WRITE, vector_size);
	cl::Buffer buffer_result(context, CL_MEM_READ_WRITE, vector_size);

	//Copy arrays A and B to device memory
	queue.enqueueWriteBuffer(buffer_temperature, CL_TRUE, 0, vector_size, &temperature[0]);
	queue.enqueueWriteBuffer(buffer_result, CL_TRUE, 0, vector_size, &result[0]);

	//Setup and execute the kernel(i.e.device code)
	cl::Kernel kernel_equal = cl::Kernel(program, "equal");
	kernel_equal.setArg(0, buffer_temperature);
	kernel_equal.setArg(1, buffer_result);

	queue.enqueueNDRangeKernel(kernel_equal, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);
	
	//Copy the result from device to host
	queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, vector_size, &result[0]);

	return 0;
}