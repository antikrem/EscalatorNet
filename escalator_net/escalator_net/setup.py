from distutils.core import setup, Extension, DEBUG

sfc_module = Extension(
        'escalator_net', 
        sources = ['pylink.cpp', 'rand_ex.cpp', 'stopwatch.cpp']
    )

setup(name = 'escalator_net neural network project', version = '1.0',
    description = 'Python Package with superfastcode C++ extension',
    ext_modules = [sfc_module]
)