#include <string>

#include <Python.h>

#include "network.hpp"

#include "pylink_helper.h"

const std::string E_NET_VERSION = "1.0";

#define E_NET_TYPE "EscalatorNetwork"

#include "_tests.hpp"

// Unpacks a PyObject with an underlying PyCapsule to a Network*
// Returns nullptr on failure
static Network<double>* extractNetwork(PyObject* item) {
	Network<double>* network;
	if ((network = (Network<double>*)PyCapsule_GetPointer(item, E_NET_TYPE))) {
		return network;
	}
	else {
		return nullptr;
	}
}

extern "C" {
	// Returns string of version
	static PyObject* getVersion(PyObject* self) {
		return PY_STRING(E_NET_VERSION);
	}
	
	// Creates a network with 
	static PyObject* Network_create(PyObject* self, PyObject* o) {
		// Check length of object
		int n = (int)PyList_Size(o);
		if (n < 0) {
			return nullptr;
		}

		std::vector<uint> nodeCount;
		for (int i = 0; i < n; i++) {
			PyObject* item = PyList_GetItem(o, i);
			nodeCount.push_back((int)PyLong_AsSize_t(item));
		}

		//return PyCapsule_New(new Network<double>(nodeCount), E_NET_TYPE, NULL);
		return PyCapsule_New(new Network<double>(nodeCount), E_NET_TYPE, NULL);
	}

	// Deletes a network
	static PyObject* Network_delete(PyObject* self, PyObject* o) {
		// Extract network
		Network<double>* network = extractNetwork(o);
		if (!network) {
			return nullptr;
		}

		delete network;

		Py_IncRef(Py_None);
		return Py_None;
	}

	// Takes a PyCapsule and gets pointer
	static PyObject* Network_get(PyObject* self, PyObject* capsule) {
		Network<double>* network;
		if ((network = (Network<double>*)PyCapsule_GetPointer(capsule, E_NET_TYPE))) {
			return PY_STRING("hello");
		}
		else {
			return nullptr;
		}
	}

	// Takes a 2 rows, first is a list of input, second is a list of output
	static PyObject* Network_addExample(PyObject* self, PyObject* args) {
		// Input/output list as py objects
		PyObject* networkPy;
		PyObject* inputPy;
		PyObject* outputPy;

		if (!PyArg_ParseTuple(args, "OOO", &networkPy, &inputPy, &outputPy)) {
			return nullptr;
		}

		// Extract network
		Network<double>* network = extractNetwork(networkPy);
		if (!network) {
			return nullptr;
		}

		// Extract size of input
		int inputLength = (int)PyList_Size(inputPy);
		int outputLength = (int)PyList_Size(outputPy);
		VMatrix<double> input = VMatrix<double>(inputLength, 1, 0.0);
		VMatrix<double> output = VMatrix<double>(outputLength, 1, 0.0);

		for (int i = 0; i < inputLength; i++) {
			PyObject* item = PyList_GetItem(inputPy, i);
			input.set(i, 0, PyFloat_AsDouble(item));
		}

		for (int i = 0; i < outputLength; i++) {
			PyObject* item = PyList_GetItem(outputPy, i);
			output.set(i, 0, PyFloat_AsDouble(item));
		}

		network->addExample(input, output);

		return networkPy;
	}

	// Trains a network with the internal input/output
	// return self on success
	static PyObject* Network_train(PyObject* self, PyObject* o) {
		// Extract network
		Network<double>* network = extractNetwork(o);
		if (!network) {
			return nullptr;
		}

		network->train(true);

		std::cout << *network << std::endl;

		return o;
	}

	static PyObject* runTests(PyObject* self) {
		tests::runAllTests();
		return PY_STRING(E_NET_VERSION);
	}
}

// Exported methods
static PyMethodDef E_NET_ENGINE_METHODS[] = {
	{ "version", (PyCFunction)getVersion, METH_NOARGS, nullptr },
	{ "run_tests", (PyCFunction)runTests, METH_NOARGS, nullptr },
	{ "Network_create", (PyCFunction)Network_create, METH_O, nullptr },
	{ "Network_delete", (PyCFunction)Network_delete, METH_O, nullptr },
	{ "Network_get", (PyCFunction)Network_get, METH_O, nullptr },
	{ "Network_addExample", (PyCFunction)Network_addExample, METH_VARARGS, nullptr },
	{ "Network_train", (PyCFunction)Network_train, METH_O, nullptr },
	{ nullptr, nullptr, 0, nullptr }
};


// Exported Module
static PyModuleDef E_NET_ENGINE_MODULE = {
	PyModuleDef_HEAD_INIT,
	"e_net_engine",
	"C Interface for Escalator Neural Network Engines",
	0,
	E_NET_ENGINE_METHODS
};


// Initialiser
PyMODINIT_FUNC PyInit_e_net_engine() {
	return PyModule_Create(&E_NET_ENGINE_MODULE);
}