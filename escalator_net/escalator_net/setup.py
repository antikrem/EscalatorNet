from distutils.core import setup, Extension, DEBUG

enet_module = Extension(
        'e_net_engine', 
        sources = ['pylink.cpp', 'rand_ex.cpp', 'stopwatch.cpp'],
        depends = ['network_wrap.py']
    )

enet_function_types = Extension(
        'function_types', ['function_types.py']
    )

enet_wrap_module = Extension(
        'escalator_net', ['network_wrap.py']
    )

setup(
        name = 'escalator_net', 
        version = '1.0',
        description = 'The Escalator Feedforward Multilayer Perceptron Project',
        author = 'Chalinda Rodrigo',
        author_email = 'chalindarodrigo@gmail.com',
        url = r'https://github.com/antikrem/EscalatorNet',
        ext_modules = [enet_module, enet_function_types, enet_wrap_module]
    )