/* General type defs
*/

#ifndef __PYLINK_HELPER__
#define __PYLINK_HELPER__

#define PY_STRING(x) Py_BuildValue("s", (std::string(x)).data())

#endif