from distutils.core import setup, Extension, DEBUG
#from Cython.Distutils import build_ext

enet_module = Extension(
        'e_net_engine', 
        sources = ['pylink.cpp', 'rand_ex.cpp', 'stopwatch.cpp'],
        depends = ['network_wrap.py']
    )

enet_wrap_module = Extension(
        'escalator_net', ['network_wrap.py']
    )

setup(
        name = 'escalator_net', 
        version = '1.0',
        description = 'The Escalator Feedforward Multilayer Perceptron Project',
        #cmdclass = {'build_ext': build_ext},
        ext_modules = [enet_module, enet_wrap_module],
    )