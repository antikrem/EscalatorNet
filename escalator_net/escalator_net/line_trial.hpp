/* Represents a layer in the network
 */
#ifndef __LINE_TRIAL__
#define __LINE_TRIAL__

#include <vector>

#include "network.hpp"

struct LineTrialSample {
	uint8 input[16];
	uint8 output[2];

	LineTrialSample(uint8* ptr, uint8 up, uint8 down) {
		alg::copy(ptr, input, 16);
		output[0] = up;
		output[1] = down;
	}
};

struct LineTrialMaster {
	std::vector<LineTrialSample> list;

	LineTrialMaster() {

	}

	void addLineTrial(std::vector<uint8> ptr, uint8 up, uint8 down) {
		list.push_back(LineTrialSample(ptr.data(), up, down));
	}

	VMatrix<double> getInput(uint inputWidth) {

		VMatrix<double> input(inputWidth, (uint)list.size(), 0.0);

		for (uint j = 0; j < list.size(); j++) {
			for (uint i = 0; i < inputWidth; i++) {
				input.set(i, j, (double)list[j].input[i]);
			}
		}

		return input;
	}

	VMatrix<double> getOutput(uint outputWidth) {

		VMatrix<double> output(outputWidth, (uint)list.size(), 0.0);

		for (uint j = 0; j < list.size(); j++) {
			for (uint i = 0; i < outputWidth; i++) {
				output.set(i, j, (double)list[j].output[i]);
			}
		}

		return output;
	}
};


 /* implementation of an xor gate
	  */
void lt_trial() {

	LineTrialMaster master;
	master.addLineTrial(
		{ 0, 0, 0, 1,
		  0, 0, 1, 0,
		  0, 1, 0, 0,
		  1, 0, 0, 0 },
		1,
		0
	);

	master.addLineTrial(
		{ 1, 0, 0, 0,
		  0, 1, 0, 0,
		  0, 0, 1, 0,
		  0, 0, 0, 1 },
		0,
		1
	);

	master.addLineTrial(
		{ 1, 1, 0, 0,
		  0, 1, 1, 0,
		  0, 0, 1, 0,
		  0, 0, 0, 1 },
		0,
		1
	);

	master.addLineTrial(
		{ 0, 0, 0, 1,
		  0, 0, 1, 0,
		  0, 1, 0, 0,
		  1, 0, 0, 0 },
		1,
		0
	);

	master.addLineTrial(
		{ 0, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0 },
		0,
		0
	);

	master.addLineTrial(
		{ 1, 1, 1, 1,
		  1, 1, 1, 1,
		  1, 1, 1, 1,
		  1, 1, 1, 1 },
		0,
		0
	);

	master.addLineTrial(
		{ 0, 0, 1, 0,
		  0, 1, 0, 0,
		  1, 0, 0, 0,
		  0, 0, 0, 0 },
		1,
		0
	);

	master.addLineTrial(
		{ 0, 1, 0, 0,
		  1, 0, 0, 0,
		  1, 0, 0, 0,
		  0, 0, 0, 0 },
		1,
		0
	);

	master.addLineTrial(
		{ 1, 1, 0, 0,
		  1, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0 },
		1,
		0
	);

	master.addLineTrial(
		{ 0, 0, 1, 0,
		  0, 0, 0, 1,
		  0, 0, 0, 1,
		  0, 0, 0, 1 },
		0,
		1
	);

	master.addLineTrial(
		{ 0, 0, 0, 0,
		  1, 1, 0, 0,
		  0, 0, 1, 0,
		  0, 0, 0, 1 },
		0,
		1
	);

	master.addLineTrial(
		{ 1, 1, 1, 1,
		  0, 0, 0, 1,
		  0, 0, 0, 1,
		  0, 0, 0, 1 },
		0,
		0
	);

	master.addLineTrial(
		{ 1, 1, 1, 1,
		  0, 0, 0, 0,
		  1, 1, 1, 1,
		  0, 0, 0, 0 },
		0,
		0
	);


	master.addLineTrial(
		{ 0, 0, 0, 0,
		  0, 0, 0, 0,
		  1, 1, 1, 1,
		  0, 0, 0, 0 },
		0,
		0
	);

	master.addLineTrial(
		{ 0, 1, 0, 1,
		  0, 0, 0, 0,
		  1, 0, 1, 0,
		  0, 0, 0, 0 },
		0,
		0
	);


	master.addLineTrial(
		{ 0, 1, 0, 0,
		  0, 1, 0, 0,
		  1, 1, 1, 1,
		  0, 1, 0, 0 },
		0,
		0
	);

	master.addLineTrial(
		{ 0, 1, 0, 0,
		  0, 1, 0, 0,
		  0, 1, 0, 0,
		  0, 1, 0, 0 },
		0,
		0
	);


	master.addLineTrial(
		{ 0, 0, 0, 1,
		  0, 0, 0, 1,
		  0, 0, 0, 1,
		  0, 0, 0, 1 },
		0,
		0
	);


	master.addLineTrial(
		{ 0, 0, 0, 1,
		  0, 0, 0, 0,
		  0, 1, 0, 0,
		  1, 0, 0, 0 },
		1,
		0
	);

	master.addLineTrial(
		{ 0, 0, 0, 0,
		  0, 0, 0, 0,
		  1, 1, 0, 0,
		  0, 0, 1, 0 },
		0,
		1
	);


	master.addLineTrial(
		{ 1, 0, 0, 0,
		  0, 1, 1, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 1 },
		0,
		1
	);

	master.addLineTrial(
		{ 1, 1, 0, 0,
		  0, 1, 1, 0,
		  0, 0, 1, 0,
		  0, 0, 0, 1 },
		0,
		1
	);


	Network<double> net(16, FunctionTypes::sigmoid, { 4, 2 }, 1.0);

	VMatrix<double> input = master.getInput(16);
	VMatrix<double> output = master.getOutput(2);

	net.addExample(input, output);
	net.train();

	VMatrix<double> trial1(
		{
			{0, 0, 0, 1,
		  0, 0, 1, 0,
		  0, 0, 0, 0,
		  1, 0, 0, 0}
		}
	);

	VMatrix<double> trial2(
		{
		{0, 0, 0, 1,
		  0, 0, 1, 1,
		  0, 0, 0, 0,
		  1, 0, 0, 0}
		}
	);

	VMatrix<double> trial3(
		{
		{1, 0, 1, 0,
		  0, 0, 1, 0,
		  0, 0, 0, 0,
		  0, 0, 1, 0}
		}
	);

	VMatrix<double> trial4(
		{
		{ 1, 0, 1, 0,
		  0, 1, 0, 1,
		  1, 0, 1, 0,
		  0, 1, 0, 1 }
		}
	);
	VMatrix<double> trial5(
		{
		{ 1, 0, 0, 1,
		  0, 1, 1, 1,
		  0, 1, 1, 0,
		  1, 0, 0, 1 }
		}
	);


	std::cout << net << std::endl;
	net.predict(trial1);
	net.predict(trial2);
	net.predict(trial3);
	net.predict(trial4);
	net.predict(trial5);
}


/* Runs all test
*/
void runLineTrial() {
	std::cout << "Line trial:" << std::endl;
	lt_trial();
	std::cout << std::endl;

}

#endif