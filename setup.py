"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

from setuptools import setup
from distutils.extension import Extension
import numpy
import platform
import os
from opendr.version import version

try:
    from Cython.Build import cythonize
    have_cython = True
except:
    cythonize = lambda x : x
    have_cython = False


# setuptools DWIM monkey-patch madness
# http://mail.python.org/pipermail/distutils-sig/2007-September/thread.html#8204
import sys
#if 'setuptools.extension' in sys.modules:
#    m = sys.modules['setuptools.extension']
#    m.Extension.__dict__ = m._Extension.__dict__

context_dir = os.path.join(os.path.dirname(__file__), 'opendr', 'contexts')

def download_osmesa():
    import os, re, zipfile
    from opendr.utils import wget
    mesa_dir = os.path.join(context_dir,'OSMesa')
    if not os.path.exists(mesa_dir):
        sysinfo = platform.uname()
        osmesa_fname = 'OSMesa.%s.%s.zip' % (sysinfo[0], sysinfo[-2])
        zip_fname = os.path.join(context_dir, osmesa_fname)
        if not os.path.exists(zip_fname):
            print(("Downloading %s" % osmesa_fname))
            # MPI url: http://files.is.tue.mpg.de/mloper/opendr/osmesa/%s
            # BL url: https://s3.amazonaws.com/bodylabs-assets/public/osmesa/%s
            wget('http://files.is.tue.mpg.de/mloper/opendr/osmesa/%s' % (osmesa_fname,), dest_fname=zip_fname)

        assert(os.path.exists(zip_fname))
        with zipfile.ZipFile(zip_fname, 'r') as z:
            for f in [x for x in z.namelist() if re.search('[ah]$', x)]:
                z.extract(f, path=context_dir)
        assert(os.path.exists(mesa_dir))


def autogen_opengl_sources():
    import os
    sources = [ os.path.join(context_dir, x) for x in ['_constants.py', '_functions.pyx'] ]
    if not all([ os.path.exists(x) for x in sources ]):
        print("Autogenerating opengl sources")
        from opendr.contexts import autogen
        autogen.main()
        for x in sources:
            assert(os.path.exists(x))


def setup_opendr(ext_modules):
    ext_modules=cythonize(ext_modules)
    try: # hack
        ext_modules[0]._convert_pyx_sources_to_lang = lambda : None
    except: pass
    setup(name='opendr',
            version=version,
            packages = ['opendr', 'opendr.contexts', 'opendr.test_dr'],
            #package_dir = {'opendr': 'opendr'},
            author = 'Matthew Loper',
            author_email = 'matt.loper@gmail.com',
            url = 'http://github.com/mattloper/opendr',
            #ext_package='opendr',
            package_data={'opendr': ['opendr/test_dr/nasa*']},
            install_requires=['Cython', 'chumpy >= 0.58', 'matplotlib'],
            description='opendr',
            ext_modules=ext_modules,
            license='MIT',

            # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
            classifiers=[
                # How mature is this project? Common values are
                #   3 - Alpha
                #   4 - Beta
                #   5 - Production/Stable
                'Development Status :: 4 - Beta',

                # Indicate who your project is intended for
                'Intended Audience :: Science/Research',
                'Topic :: Multimedia :: Graphics :: 3D Rendering',

                # Pick your license as you wish (should match "license" above)
                'License :: OSI Approved :: MIT License',

                # Specify the Python versions you support here. In particular, ensure
                # that you indicate whether you support Python 2, Python 3 or both.
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.7',

                'Operating System :: MacOS :: MacOS X',
                'Operating System :: POSIX :: Linux'
            ],

          )


def mesa_ext():
    libraries = ['OSMesa', 'GL', 'GLU']
    extra_args = []
    if platform.system()=='Darwin': # deprecated, probably don't need osmesa libs on mac
        libraries.append('talloc')
        extra_args.append('-Qunused-arguments')
    else:
        extra_args.append('-lstdc++')
    return Extension("opendr.contexts.ctx_mesa", ['opendr/contexts/ctx_mesa.pyx'] if have_cython else ['opendr/contexts/ctx_mesa.c'],
                        language="c",
                        library_dirs=['opendr/contexts/OSMesa/lib'],
                        depends=['opendr/contexts/_constants.py'],
                        define_macros = [('__OSMESA__', 1)],
                        include_dirs=['.', numpy.get_include(), 'opendr/contexts/OSMesa/include'],
                        libraries=libraries,
                        extra_compile_args=extra_args,
                        extra_link_args=extra_args)

def mac_ext():
    return Extension("opendr.contexts.ctx_mac", ['opendr/contexts/ctx_mac.pyx', 'opendr/contexts/ctx_mac_internal.c'] if have_cython else ['opendr/contexts/ctx_mac.c', 'opendr/contexts/ctx_mac_internal.c'],
        language="c",
        depends=['opendr/contexts/_constants.py', 'opendr/contexts/ctx_mac_internal.h'],
        include_dirs=['.', numpy.get_include()],
        extra_compile_args=['-Qunused-arguments'],
        extra_link_args=['-Qunused-arguments'])


def main():
    from opendr.contexts.fix_warnings import fix_warnings
    fix_warnings()

    # Get osmesa and some processed files ready
    download_osmesa()
    autogen_opengl_sources()

    # Get context extensions ready & build
    if platform.system() == 'Darwin':
        setup_opendr([mac_ext()])
    else:
        setup_opendr([mesa_ext()])


if __name__ == '__main__':
    main()

