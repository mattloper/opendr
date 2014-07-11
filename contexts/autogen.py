#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""


import sys
import os
import re
from os.path import split, join
import platform


def main():
    functions_fname = join(split(__file__)[0], '_functions.pyx')
    constants_fname = join(split(__file__)[0], '_constants.py')

    from os.path import exists
    if exists(functions_fname) and exists(constants_fname):
        return

    constants = ''
    functions = 'cdef extern from "gl_includes.h":\n'
    #extfunctions = 'cdef extern from "GL/glext.h":\n'
    extfunctions = ''
    
    text = open(split(__file__)[0] + '/OSMesa/include/GL/gl.h').read().replace('const', '')
    text += '\ntypedef char GLchar;\n'
    defines = re.findall('#define GL_(.*?)\s+(.*?)\n', text, flags=re.DOTALL)
    defines += re.findall('#define GL_(.*?)\s+(.*?)\n', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    funcs = re.findall('GLAPI (\w+) GLAPIENTRY gl(\w+)\s*\((.*?)\);', text, flags=re.DOTALL)

    extfuncs = re.findall('GLAPI (void) APIENTRY gl(Generate\w*)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(GenBuffer\w*)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(BindBuffer\w*)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(BufferData)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)

    extfuncs += re.findall('GLAPI (void) APIENTRY gl(GenVertexArrays)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(BindVertexArray)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(VertexAttrib4fv)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(DrawArrays)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(DeleteVertexArrays)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(ClearBufferfv)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(EnableVertexAttribArray)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(VertexAttribPointer)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)

    extfuncs += re.findall('GLAPI (void) APIENTRY gl(ValidateProgram)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(GetProgramInfoLog)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(DeleteProgram)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (GLuint) APIENTRY gl(CreateProgram)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (GLuint) APIENTRY gl(CreateShader)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(ShaderSource)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(CompileShader)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(AttachShader)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(LinkProgram)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(UseProgram)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (GLint) APIENTRY gl(GetUniformLocation)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(Uniform1i)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(Uniform4f)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(Uniform1f)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (GLint) APIENTRY gl(GetAttribLocation)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(BindFragDataLocation)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    extfuncs += re.findall('GLAPI (void) APIENTRY gl(UniformMatrix4fv)\s*\((.*?)\);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)



    typedefs = re.findall('typedef (\w[^\n]+GL\w+);', text, flags=re.DOTALL)
    typedefs += re.findall('typedef (\w[^\n]+GL\w+ptr);', open(split(__file__)[0] + '/OSMesa/include/GL/glext.h').read().replace('const', ''), flags=re.DOTALL)
    constants_all = ''
    for define in defines:
        if 'INVALID_INDEX' not in define[0] and 'TIMEOUT_IGNORED' not in define[0]:
            constants += 'GL_%s = %s\n' % (define[0], define[1])
            constants_all += "'GL_%s'," % (define[0])
        
    for typedef in typedefs:
        functions += '\tctypedef %s\n' % (typedef,)
        
    for func in funcs:
        args = func[2]
        if args.strip() == 'void':
            args = ''
        if 'GLprogramcallbackMESA' in args:
            continue
        if 'GLcharARB' in args:
            continue
        if 'GLhandleARB' in args or 'GLhandleARB' in func[0]:
            continue
        args = args.replace(' in,', 'inp,')
        
        nogil = '' if ('*' in args and func[1] != 'ReadPixels') else 'nogil'
        functions += '\tcdef %s _gl%s "gl%s"(%s) %s\n' % (func[0], func[1], func[1], args, nogil)

    for func in extfuncs:
        args = func[2]
        if args.strip() == 'void':
            args = ''
        if 'GLprogramcallbackMESA' in args:
            continue
        if 'GLcharARB' in args:
            continue
        if 'GLhandleARB' in args or 'GLhandleARB' in func[0]:
            continue
        args = args.replace(' in,', 'inp,')
        nogil = '' if '*' in args else 'nogil'
        extfunctions += '\tcdef %s _gl%s "gl%s"(%s) %s\n' % (func[0], func[1], func[1], args, nogil)
    # for t in text.split('\n'):    
    #     if t.startswith('#define GL_'):
    #         m = re.match(r"#define (GL_\w+)\s*(\w+)",t)
    #         constants += '%s = %s\n' % (m.group(1), m.group(2))
    #         exports.append(m.group(1))
    #         
    #     if t.startswith('typedef'):
    #         try:
    #             m = re.match(r"typedef (\w+\s.*GL\w+);",t)                
    #             functions += '\tctypedef %s\n' % (m.group(1))
    #         except: pass
    #         
    #     if 'ReadPixels' in t:
    #         import pdb; pdb.set_trace()
    #     if t.startswith('GLAPI void GLAPIENTRY gl'):
    #         try:
    #             m = re.match(r"GLAPI (\w+) GLAPIENTRY gl(\w+)\((.*)\);", t, flags=re.dotall)
    #             returntype = m.group(1)
    #             funcname = m.group(2)
    #             fargs = m.group(3)
    # 
    #             if fargs.strip() == 'void':
    #                 fargs = ''
    #             if 'GLprogramcallbackMESA' not in fargs:
    #                 functions += '\tcdef %s _gl%s "gl%s"(%s)\n' % (returntype, funcname, funcname, fargs)
    #         except: pass
    # 
    # text = open(split(__file__)[0] + '/include/GL/glu.h').read().replace('const','')
    # for t in text.split('\n'):    
    #     if t.startswith('#define GLU_'):
    #         m = re.match(r"#define (GLU_\w+)\s*(\w+)",t)
    #         constants += '%s = %s\n' % (m.group(1), m.group(2))
    #         exports.append(m.group(1))

    #GLAPI void GLAPIENTRY glTexCoord4d( GLdouble s, GLdouble t, GLdouble r, GLdouble q );

    with open(functions_fname, 'w') as fp:
        fp.write(functions)
        fp.write(extfunctions)

    constants_all = "__all__ = [%s]" % (constants_all,)

    constants = constants_all + '\n' + constants

    with open(constants_fname, 'w') as fp:
        fp.write(constants)


if __name__ == '__main__':
    main()

