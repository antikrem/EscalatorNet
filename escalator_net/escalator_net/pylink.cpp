#include <string>

#include <Python.h>

#include "network.hpp"

#include "pylink_helper.h"

const std::string E_NET_VERSION = "0.9";

#define E_NET_TYPE "EscalatorNetwork"

extern "C" {
	// Returns string of version
	static PyObject* getVersion(PyObject* self) {
		return PY_STRING(E_NET_VERSION);
	}
}

// Exported methods
static PyMethodDef E_NET_ENGINE_METHODS[] = {
	{ "version", (PyCFunction)getVersion, METH_NOARGS, nullptr },
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