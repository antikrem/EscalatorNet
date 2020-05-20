#include <Windows.h>
#include <cmath>
#include <Python.h>

#include "_tests.hpp"

const double e = 2.7182818284590452353602874713527;

double sinh_impl(double x) {
	return (1 - pow(e, (-2 * x))) / (2 * pow(e, -x));
}

double cosh_impl(double x) {
	return (1 + pow(e, (-2 * x))) / (2 * pow(e, -x));
}

PyObject* tanh_impl(PyObject *, PyObject* o) {
	double x = PyFloat_AsDouble(o);
	double tanh_x = sinh_impl(x) / cosh_impl(x);
	return PyFloat_FromDouble(tanh_x);
}
extern "C" {

	// Runs all inbuilt tests
	PyObject* run_tests(PyObject *, PyObject* o) {
		rand_ex::reset();
		tests::runAllTests();
		return Py_BuildValue("");
	}

}

// Exported methods
static PyMethodDef ESCALATOR_NET_METHODS[] = {
	{ "run_tests", (PyCFunction)run_tests, METH_O, nullptr },
	{ nullptr, nullptr, 0, nullptr }
};


// Exported Module
static PyModuleDef ESCALATOR_NET_MODULE = {
	PyModuleDef_HEAD_INIT,
	"escalator_net",
	"Provides a feed forward multi-layer perceptron",
	0,
	ESCALATOR_NET_METHODS   
};


// Initialiser
PyMODINIT_FUNC PyInit_escalator_net() {
	return PyModule_Create(&ESCALATOR_NET_MODULE);
}