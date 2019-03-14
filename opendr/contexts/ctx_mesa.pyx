"""
Author(s): Matthew Loper

See LICENCE.txt for licensing information.
"""

include "ctx_base.pyx"
    
   
cdef extern from "GL/osmesa.h":
    ctypedef struct osmesa_context: 
        pass
    ctypedef osmesa_context* OSMesaContext
    cdef OSMesaContext _OSMesaCreateContext "OSMesaCreateContext"( GLenum format, OSMesaContext sharelist )
    cdef GLboolean _OSMesaMakeCurrent "OSMesaMakeCurrent"(OSMesaContext ctx, void *buffer, GLenum type, GLsizei width, GLsizei height )
    cdef void _OSMesaDestroyContext "OSMesaDestroyContext"( OSMesaContext ctx )
    
cdef extern from "GL/glu.h":
    cdef void _gluLookAt "gluLookAt"(GLdouble eyeX, GLdouble eyeY, GLdouble eyeZ, GLdouble centerX, GLdouble centerY, GLdouble centerZ, GLdouble upX, GLdouble upY, GLdouble upZ)
    cdef void _gluPerspective "gluPerspective"(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar)
    cdef void _gluUnProject "gluUnProject"(GLdouble  	winX,
 	GLdouble  	winY,
 	GLdouble  	winZ,
 	GLdouble *  	model,
 	GLdouble *  	proj,
 	GLint *  	view,
 	GLdouble*  	objX,
 	GLdouble*  	objY,
 	GLdouble*  	objZ)
    cdef void _gluProject "gluProject"(	GLdouble  	objX,
 	GLdouble  	objY,
 	GLdouble  	objZ,
 	GLdouble *  	model,
 	GLdouble *  	proj,
 	GLint *  	view,
 	GLdouble*  	winX,
 	GLdouble*  	winY,
 	GLdouble*  	winZ)

    

cdef class OsContextRaw(OsContextBase):
    cdef OSMesaContext ctx
    cdef np.ndarray image_internal
    cdef GLenum typ_internal

    def __init__(self, w, h, format=GL_RGB, typ=GL_UNSIGNED_BYTE):
        self.format = format
        self.ctx = _OSMesaCreateContext(GL_RGBA, NULL)
        self.typ = typ
        self.typ_internal = GL_UNSIGNED_BYTE

        # If you change this stuff, test it in Linux and on Mac!
        # I couldn't get RGB to work as an internal format for
        # OSMesa (only RGBA worked for me), and we're currently
        # not compiled (afaik) for 32-bit. But the problems are 
        # a bit subtle, so test, test, test!
        #
        # -Matt
        #
        if self.typ == GL_FLOAT:
            self.image = np.zeros((h,w,3), dtype=np.float32)
        elif self.typ == GL_UNSIGNED_BYTE:
            self.image = np.zeros((h,w,3), dtype=np.uint8)
        if self.typ_internal == GL_UNSIGNED_BYTE:
            self.image_internal = np.zeros((h,w,4), dtype=np.uint8)
        elif self.typ_internal == GL_FLOAT:
            self.image_internal = np.zeros((h,w,4), dtype=np.float32)

        self.w = w
        self.h =h
        self.depth = np.zeros_like(self.image[:,:,0], dtype=np.float32)
        
    def __del__(self):
        _OSMesaDestroyContext(self.ctx)
    def MakeCurrent(self):
        _OSMesaMakeCurrent(self.ctx, <void*>self.image_internal.data, self.typ_internal, self.w, self.h)


class OsContext(OsContextRaw):
    pass