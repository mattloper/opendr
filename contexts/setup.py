"""
Copyright (C) 2013
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['build_contexts']

from os.path import join, split, exists
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import re
import platform
from opendr.utils import wget
import zipfile


def lf(fname):
    return join(split(__file__)[0], fname)

def is_osx():
    return bool(re.search('darwin', platform.platform(), re.I))

def ctx_mesa_libraries():
    libraries = ['OSMesa', 'GL', 'GLU']
    if is_osx():
        libraries.append('talloc')
    else:
        libraries.append('stdc++')
    return libraries

def build_contexts():
    if not exists(lf('_constants.py')) or not exists(lf('_functions.pyx')):
        import autogen
        autogen.main()
        assert(exists(lf('_constants.py')) and exists(lf('_functions.pyx')))
    ctx_mesa_extension = Extension("ctx_mesa", [lf('ctx_mesa.pyx')],
                        language="c",
                        library_dirs=[lf('OSMesa/lib')],
                        depends=[lf('_functions.pyx'), lf('_constants.py'), lf('ctx_base.pyx')],
                        define_macros = [('__OSMESA__', 1)],
                        libraries=ctx_mesa_libraries())

    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [ctx_mesa_extension],
        include_dirs = ['.', numpy.get_include(), lf('OSMesa/include')],
    )

    if platform.system()=='Darwin':
        ctx_mac_extension = Extension("ctx_mac", [lf('ctx_mac.pyx'), lf('ctx_mac_internal.c')],
                        language="c",
                        depends=[lf('_functions.pyx'), lf('_constants.py'), lf('ctx_base.pyx'), lf('ctx_mac_internal.h')],
                        extra_compile_args=['-framework', 'OpenGL'],
                        extra_link_args=['-framework', 'OpenGL'])


        setup(
            cmdclass = {'build_ext': build_ext},
            ext_modules = [ctx_mac_extension],
            include_dirs = ['.', numpy.get_include()]
            )


def download_osmesa():
    curdir = '.'
    mesadir = join(curdir,'OSMesa')
    if not exists(mesadir):
        sysinfo = platform.uname()
        osmesa_fname = 'OSMesa.%s.%s.zip' % (sysinfo[0], sysinfo[-2])
        dest_fname = '%s/%s' % (curdir, osmesa_fname,)
        if not exists(dest_fname):
            wget('http://files.is.tue.mpg.de/mloper/opendr/osmesa/%s' % (osmesa_fname,), dest_fname=dest_fname)
        assert(exists(dest_fname))

        with zipfile.ZipFile(dest_fname, 'r') as z:
            for f in filter(lambda x: re.search('[ah]$', x), z.namelist()):
                z.extract(f, path='.')
        assert(exists(mesadir))


if __name__ == '__main__':
    download_osmesa()
    build_contexts()
