"""
Author(s): Matthew Loper

See LICENCE.txt for licensing information.
"""

include "ctx_base.pyx"
   
cdef extern from "ctx_mac_internal.h":
    cdef void *create_context (unsigned int imageWidth, unsigned int imageHeight, GLenum typ)
    cdef void set_current(void *ctx)
    cdef void release_context(void *ctx)


cdef extern from "OpenGL/glu.h":
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
    cdef void _gluProject "gluProject"(
        GLdouble  	objX,
        GLdouble  	objY,
        GLdouble  	objZ,
        GLdouble *  	model,
        GLdouble *  	proj,
        GLint *  	view,
        GLdouble*  	winX,
        GLdouble*  	winY,
        GLdouble*  	winZ)
   
   


cdef class OsContextRaw(OsContextBase):
    
    cdef void* ctx

    def __init__(self, w, h, format=GL_RGB, typ=GL_UNSIGNED_BYTE):
    
        self.format = format

        self.ctx = create_context (w, h, typ)
        self.typ = typ
        if typ == GL_UNSIGNED_BYTE:
            self.image = np.zeros((h,w,3), dtype=np.uint8)
        elif typ == GL_FLOAT:
            self.image = np.zeros((h,w,3), dtype=np.float32)
        else:
            raise Exception('Need GL_FLOAT or GL_UNSIGNED_BYTE for typ')
      
        self.w = w
        self.h =h
        self.depth = np.zeros_like(self.image[:,:,0], dtype=np.float32)
        
    def __del__(self): 
        release_context(self.ctx)

    def MakeCurrent(self):
        set_current(self.ctx)

class OsContext(OsContextRaw):
    pass