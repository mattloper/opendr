#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing information.
"""

import numpy as np
cimport numpy as np
np.import_array()
from libc.stdio cimport printf

ctypedef size_t	ptrdiff_t


include "_functions.pyx"
from _constants import *


def mc(func):
    def with_make_current(self, *args, **kwargs):
        self.MakeCurrent()
        return func(self, *args, **kwargs)
    return with_make_current


cdef class OsContextBase(object):
    cdef np.ndarray image
    cdef np.ndarray depth
    cdef GLenum format
    cdef int w
    cdef int h
    cdef GLenum typ
        
    def MakeCurrent(self):
        raise NotImplemented
    
    @property
    def width(self):
        return self.w
    
    @property
    def height(self):
        return self.h


    @mc
    def VertexAttrib4fv(self, GLuint index, np.ndarray[float, ndim=1] v):
        _glVertexAttrib4fv(index, &v[0])


    #GLAPI void GLAPIENTRY glGetIntegerv( GLenum pname, GLint *params );
    @mc
    def GetInteger(self, GLenum pname):
        cdef int result
        _glGetIntegerv(pname, &result)
        return result

    # GLAPI void APIENTRY glEnableVertexAttribArray (GLuint index);
    @mc
    def EnableVertexAttribArray(self, GLuint index) : _glEnableVertexAttribArray(index)

    # @mc
    # def DeleteVertexArrays(self, GLsizei n, const GLuint *arrays): _glDeleteVertexArrays(n, arrays)

    @mc
    def GenVertexArrays(self, GLsizei n):
        cdef np.ndarray[unsigned int, ndim=1] arr
        arr = np.zeros(n, dtype=np.uint32)
        _glGenVertexArrays(n, &arr[0])
        return arr

    #GLAPI void APIENTRY glBindFragDataLocation (GLuint program, GLuint color, const GLchar *name);
    @mc
    def BindFragDataLocation(self, GLuint program, GLuint color, GLchar * name):
        _glBindFragDataLocation(program, color, name)

    #  GLAPI void APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
    @mc
    def VertexAttribPointerf(self, GLuint index, GLint size, GLboolean normalized, GLsizei stride):
        _glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, 4*size, NULL)

    #  GLAPI void APIENTRY glVertexAttribPointer (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
    @mc
    def VertexAttribPointer(self, GLuint index, GLint size, GLenum typ, GLboolean normalized, GLsizei stride):

        if typ == GL_FLOAT:
            sz = 4
        elif typ == GL_DOUBLE:
            sz = 8
        else:
            raise Exception('NYI')
        _glVertexAttribPointer(index, size, typ, GL_FALSE, sz*size, NULL)

    #GLAPI void APIENTRY glClearBufferfv (GLenum buffer, GLint drawbuffer, const GLfloat *value);
    @mc
    def ClearBufferfv(self, GLenum buffer, GLint drawbuffer, np.ndarray[float, ndim=1] value):
        _glClearBufferfv(buffer, drawbuffer, <GLfloat *>value.data)


    #GLAPI GLint APIENTRY glGetAttribLocation (GLuint program, const GLchar *name);
    @mc
    def GetAttribLocation(self, GLuint program, char *name):
        return _glGetAttribLocation(program, name)

    @mc
    def DrawArrays(self, GLenum mode, GLint first, GLsizei count): _glDrawArrays(mode, first, count)

    @mc
    def DeleteProgram(self, GLuint program): _glDeleteProgram(program)

    @mc
    def RasterPos2i(self, x, y): _glRasterPos2i(x,y)
    @mc
    def PushMatrix(self): _glPushMatrix()
    @mc
    def PopMatrix(self): _glPopMatrix()
    @mc
    def BufferData(self, target, np.ndarray data, GLenum usage) : _glBufferData(target, data.nbytes, data.data, usage)

    @mc
    def LineWidth(self, GLfloat width):
        _glLineWidth(width)

    @mc
    def BindBuffer(self, GLenum target, GLuint buffer): _glBindBuffer(target, buffer)
    
    @mc
    def GenBuffers(self, n):
        cdef np.ndarray[unsigned int, ndim=1] ids
        ids = np.zeros(n, dtype=np.uint32)
        _glGenBuffers(n, &ids[0] )
        return ids
    @mc
    def Viewport(self, x0, y0, w, h): _glViewport(x0, y0, w, h)    
    @mc
    def uPerspective(slef, fovy, aspect, zNear, zFar): _gluPerspective(fovy, aspect, zNear, zFar)
    @mc
    def uLookAt(self, eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ): _gluLookAt(eyeX, eyeY, eyeZ, centerX, centerY, centerZ, upX, upY, upZ)
    @mc
    def Ortho(self, left, right, bottom, top, near=-1, far=1): _glOrtho(left, right, bottom, top, near, far)
    @mc
    def Begin(self, mode): _glBegin(mode)
    @mc
    def End(self): _glEnd()
    @mc
    def Vertex3f(self, x,y,z): _glVertex3f(x,y,z)
    @mc
    def Vertex3i(self, x,y,z): _glVertex3i(x,y,z)
    @mc
    def MatrixMode(self, mode): _glMatrixMode(mode)
    @mc
    def LoadIdentity(self): _glLoadIdentity()
    @mc
    def BlendFunc(self, sfactor, dfactor): _glBlendFunc(sfactor, dfactor)
    @mc
    def Frustum(self, left, right, bottom, top, near, far): _glFrustum(left, right, bottom, top, near, far)
    @mc
    def Rotatef(self, degrees, x, y, z): _glRotatef(degrees, x, y, z)
    @mc
    def MultMatrixf(self, np.ndarray[float, ndim=2, mode="fortran"] mtx not None): _glMultMatrixf(<GLfloat *>(&mtx[0,0]))
    @mc
    def Enable(self, what): _glEnable(what)
    @mc
    def PolygonMode(self, face, mode): _glPolygonMode(face, mode)
    @mc
    def PolygonOffset(self, factor, units): _glPolygonOffset(factor, units)
    @mc
    def Disable(self, what): _glDisable(what)
    @mc
    def Clear(self, mask): _glClear(mask)
    @mc
    def EnableClientState(self, cap): _glEnableClientState(cap)    
    @mc
    def ClearColor(self, red, green, blue, alpha): _glClearColor(red, green, blue, alpha)    
    @mc
    def VertexPointer(self, np.ndarray[double, ndim=2] pointer not None): _glVertexPointer(pointer.shape[1], GL_DOUBLE, 0, &pointer[0,0])
    @mc
    def VertexPointer0(self): _glVertexPointer(3, GL_DOUBLE, 0, NULL)
    @mc
    def ColorPointer0(self): _glColorPointer(3, GL_DOUBLE, 0, NULL)
    @mc
    def ColorPointerd(self, np.ndarray[double, ndim=2] pointer not None): _glColorPointer(3, GL_DOUBLE, 0, &pointer[0,0])
    @mc
    def ColorPointerub(self, np.ndarray[unsigned char, ndim=2] pointer not None): _glColorPointer(3, GL_UNSIGNED_BYTE, 0, &pointer[0,0])
    @mc
    def DrawElements(self, mode, np.ndarray[unsigned int, ndim=1] indices not None): _glDrawElements(mode, indices.size, GL_UNSIGNED_INT, &indices[0])

    #GLAPI void GLAPIENTRY glDrawElements( GLenum mode, GLsizei count,
    #                                  GLenum type, const GLvoid *indices );
    @mc
    def DrawElements2(self, GLenum mode, GLsizei count, GLenum typ):
        _glDrawElements(mode, count, typ, NULL)


    @mc
    def ClearDepth( self, GLclampf depth): _glClearDepth(depth)


    @mc
    def DepthFunc(self, GLenum typ): _glDepthFunc(typ)    

    @mc
    def DrawPixels(self, width, height, fmt, typ, np.ndarray[float, ndim=3] data not None): _glDrawPixels(width, height, fmt, typ, &data[0,0,0])    
    @mc    
    def Color3d(self, red, green, blue): _glColor3d(red, green, blue)
    @mc
    def Color3ub(self, red, green, blue): _glColor3ub(red, green, blue)
    @mc
    def Vertex3d(self, x, y, z): _glVertex3d(x,y,z)
    @mc
    def Finish(self): _glFinish()
    @mc
    def Flush(self): _glFlush()
    @mc
    def Hint(self, target, mode): _glHint(target, mode)
    @mc
    def BindTexture(self, target, texture): _glBindTexture(target, texture)
    @mc
    def PixelStorei(self, pname, param): _glPixelStorei(pname, param)    
    @mc
    def TexImage2Dub(self, target, level, internalFormat, width, height, border, format, np.ndarray[unsigned char, ndim=1] data): _glTexImage2D(target, level, internalFormat, width, height, border, format, GL_UNSIGNED_BYTE, <GLubyte *>(&data[0]))
    @mc
    def TexImage2Df(self, target, level, internalFormat, width, height, border, format, np.ndarray[float, ndim=1] data): _glTexImage2D(target, level, internalFormat, width, height, border, format, GL_FLOAT, <GLfloat *>(&data[0]))
    @mc
    def TexParameterf(self, target, pname, param): _glTexParameterf(target, pname, param)
    @mc
    def DeleteTextures(self, np.ndarray[unsigned int, ndim=1] textures) : _glDeleteTextures(textures.size, <GLuint *>(&textures[0]))
    @mc
    def GenerateMipmap(self, target): _glGenerateMipmap(target) 
    @mc
    def GenTextures(self, n, np.ndarray[unsigned int, ndim=1] textures): _glGenTextures(n, <GLuint *>(&textures[0]))
    @mc
    def TexEnvf(self, target, pname, param): _glTexEnvf(target, pname, param)
    @mc
    def TexCoordPointerf(self, size, stride, np.ndarray[float, ndim=1] ptr): _glTexCoordPointer(size, GL_FLOAT, stride, <GLvoid *>(&ptr[0]))
    @mc
    def TexCoordPointer0(self, size, typ, stride): _glTexCoordPointer(size, typ, stride, <GLvoid *>NULL)
    @mc
    def DisableClientState(self, cap): _glDisableClientState(cap)
    @mc
    def GetError(self): return _glGetError()

    # void glValidateProgram(	GLuint program);



    @mc
    def getImage(self):
        _glReadPixels( 0, 0, self.w, self.h, self.format, self.typ, <GLvoid*>self.image.data)
        return self.image


    @mc
    def getDepth(self):
        return self.getDepthCloud(camera_space=True)[:,:,2].copy()

    @mc
    def getDepthCloud(self, artificial_depth=None, camera_space=False):
        # fast but doesn't work
        # cdef np.ndarray[float, ndim=2] mtx
        # mtx = np.zeros((4,4), order='F', dtype=np.float32)
        # _glGetFloatv(GL_PROJECTION_MATRIX, <GLfloat*>(&mtx[0,0]))
        # a = mtx[2,2]
        # b = mtx[2,3]
        # 
        # near = b/(a-1.)
        # far = b/(a+1.)
        # print 'near is %.2f, far is %.2f' % (near, far)
        # 
        # _glReadPixels( 0, 0, self.w, self.h, GL_DEPTH_COMPONENT, GL_FLOAT, <GLvoid*>self.depth.data)
        # self.depth = ((-self.depth+1.)/2. * (far-near)) + near
        # #self.depth = near / (1. - self.depth * (1. - (near/far)))
        # return self.depth

        ###
        
        # Fast enough, works
        _glReadPixels( 0, 0, self.w, self.h, GL_DEPTH_COMPONENT, GL_FLOAT, <GLvoid*>self.depth.data)

        winX = np.arange(self.w*self.h, dtype=np.float32) % self.w
        winY = np.arange(self.w*self.h, dtype=np.float32) // self.w
        winZ = self.depth.ravel()
        
        cdef int view[4]
        _glGetIntegerv(GL_VIEWPORT, view)        
        
        cdef np.ndarray[float, ndim=1] x2
        cdef np.ndarray[float, ndim=1] y2
        cdef np.ndarray[float, ndim=1] z2
        x2 = 2 * (winX - view[0]) / view[2] - 1
        y2 = 2 * (winY - view[1]) / view[3] - 1
        z2 = winZ * 2. - 1
        
        cdef np.ndarray[float, ndim=2] P
        P = np.zeros((4,4), order='F', dtype=np.float32)
        _glGetFloatv(GL_PROJECTION_MATRIX, <GLfloat*>(&P[0,0]))
        
        cdef np.ndarray[float, ndim=2] M
        M = np.asarray(np.eye(4), dtype=np.float32, order='F')

        if camera_space or artificial_depth is not None:
            M[:,1:3] *= -1 # to get into opencv coordinate system
        else:
            _glGetFloatv(GL_MODELVIEW_MATRIX, <GLfloat*>(&M[0,0]))


        cdef np.ndarray[float, ndim=2] PM
        PM = P.dot(M)
        
        cdef np.ndarray[float, ndim=2] objCoords
        objCoords = np.linalg.inv(PM).dot(np.vstack((x2,y2,z2, np.ones_like(z2))))
        xWanted = (objCoords[0,:] / objCoords[3,:]).reshape((self.h, self.w)).astype(np.float64)
        yWanted = (objCoords[1,:] / objCoords[3,:]).reshape((self.h, self.w)).astype(np.float64)
        zWanted = (objCoords[2,:] / objCoords[3,:]).reshape((self.h, self.w)).astype(np.float64)

        result = np.dstack([xWanted, yWanted, zWanted])
        if artificial_depth != None:
            result = result / np.atleast_3d(result[:,:,2])
            result = result * np.atleast_3d(artificial_depth)

        return result

        # works but is slow
        # cdef int viewport[4]
        # #cdef double modelview[16]
        # cdef double projection[16]
        # cdef float winX, winY, winZ
        # cdef double posX, posY, posZ
        #  
        # #_glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
        # cdef np.ndarray[np.float64_t, ndim=2] modelview = np.eye(4)
        # _glGetDoublev( GL_PROJECTION_MATRIX, projection );
        # _glGetIntegerv( GL_VIEWPORT, viewport );
        #      
        # cdef np.ndarray[np.float64_t, ndim=1] mtx = np.zeros((self.width*self.height))
        # cdef double dstX, dstY, dstZ
        # 
        # _glReadPixels( 0, 0, self.w, self.h, GL_DEPTH_COMPONENT, GL_FLOAT, <GLvoid*>self.depth.data)
        #      
        # for winX in range(self.width):
        #     for winY in range(self.height):
        #         coord = int(winX+winY*self.width)
        #         winZ = self.depth.ravel()[coord]        
        #         _gluUnProject( winX, winY, winZ, &modelview[0,0], projection, viewport, &dstX, &dstY, &dstZ)
        #         mtx[coord] = -dstZ
        #     
        # mtx2 = mtx.copy().reshape((self.height, self.width))
        # return mtx2
        
    
    @mc
    def getDepthCloud_old(self, depth=None):
        cdef int viewport[4]
        cdef double modelview[16]
        cdef double projection[16]
        cdef float winX, winY, winZ
        cdef double winX2, winY2, winZ2
        cdef double posX, posY, posZ
        cdef double camposX, camposY, camposZ
        cdef np.ndarray[np.float64_t, ndim=2] modelview_eye = np.eye(4)
        #cdef double modelview_eye[16]
         
        _glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
        _glGetDoublev( GL_PROJECTION_MATRIX, projection );
        _glGetIntegerv( GL_VIEWPORT, viewport );
     
        #winX = (float)x;
        #winY = (float)viewport[3] - (float)y;

        cdef np.ndarray[np.float64_t, ndim=2] mtx = np.zeros((self.width*self.height,3))
        cdef double dstX, dstY, dstZ
        cdef np.ndarray[float, ndim=2] proj

        cdef np.ndarray[np.float64_t, ndim=3] rays = np.zeros((self.height, self.width, 3))
        if depth is None:
            _glReadPixels( 0, 0, self.w, self.h, GL_DEPTH_COMPONENT, GL_FLOAT, <GLvoid*>self.depth.data)
         
            for winX in range(self.width):
                for winY in range(self.height):
                    coord = int(winX+winY*self.width)
                    winZ = self.depth.ravel()[coord]        
                    _gluUnProject( winX, winY, winZ, modelview, projection, viewport, &dstX, &dstY, &dstZ)
                    mtx[coord,0] = dstX
                    mtx[coord,1] = dstY
                    mtx[coord,2] = dstZ
        else:
            proj = np.zeros((4,4), order='F', dtype=np.float32)
            _glGetFloatv(GL_PROJECTION_MATRIX, <GLfloat*>(&proj[0,0]))
            a = proj[2,2]
            b = proj[2,3]
            
            near = b/(a-1.)
            far = b/(a+1.)

            #_gluProject(0., 0., -1., &modelview_eye[0,0], projection, viewport, &winX2, &winY2, &winZ2)
            #_gluUnProject( winX, winY, depth[winY, winX]/winZ2, modelview, projection, viewport, &dstX, &dstY, &dstZ)
            for winX in range(self.width):
                for winY in range(self.height):
                    coord = int(winX+winY*self.width)
                    #_gluUnProject( winX, winY, ((depth[winY, winX]-near)/(far-near)), modelview, projection, viewport, &dstX, &dstY, &dstZ)                    
                    _gluUnProject( winX, winY, 1., modelview, projection, viewport, &dstX, &dstY, &dstZ)
                    dstXfar, dstYfar, dstZfar = dstX, dstY, dstZ
                    _gluUnProject( winX, winY, 0., modelview, projection, viewport, &dstX, &dstY, &dstZ)
                    dstXnear, dstYnear, dstZnear = dstX, dstY, dstZ
                    rays[int(winY), int(winX),:] = np.array([dstXfar-dstXnear, dstYfar-dstYnear, dstZfar-dstZnear])
            rays = rays / np.atleast_3d(np.sqrt(np.sum(rays**2, axis=2)))
            optical_axis = np.mean(np.mean(rays, axis=0), axis=0)
            optical_axis = optical_axis / np.linalg.norm(optical_axis)
            
            dp = rays.copy()
            dp[:,:,0] *= optical_axis[0]
            dp[:,:,1] *= optical_axis[1]
            dp[:,:,2] *= optical_axis[2]
            dp = np.sum(dp, axis=2)
            rays /= np.atleast_3d(dp)
            rays *= np.atleast_3d(depth)
            
            #import matplotlib.pyplot as plt
            #plt.imshow(rays*.5 + .5)
            #plt.draw()
            #print rays
            #print np.max(rays.ravel())
            #print np.min(rays.ravel())
            #print 'sleeping'
            #import time
            #time.sleep(10000)
            
            mtx = rays.reshape((-1,3))
            
            #mtx[coord,0] = dstX
            #mtx[coord,1] = dstY
            #mtx[coord,2] = dstZ
            

                    
            #mtx[:,0] -= camposX
            #mtx[:,1] -= camposY
            #mtx[:,2] -= camposZ
            #norms = np.sqrt(np.sum(mtx**2, axis=1))
            #mtx /= norms
            #mtx *= depth
            #mtx[:,0] += camposX
            #mtx[:,1] += camposY
            #mtx[:,2] += camposZ
            #mtx = mtx / col(mtx[:,2])
            #mtx = mtx * col(depth.ravel())
            #mtx[:,0:2] = mtx[:,0:2] / mtx[:,2].reshape((-1,1)) * depth.reshape((-1,1))
            #mtx[:,2] = depth.ravel()
            
     
        return mtx


    @mc
    def CreateProgram(self): return _glCreateProgram()

    @mc
    def CreateShader(self, GLenum typ): return _glCreateShader(typ)

    #GLAPI void APIENTRY glShaderSource (GLuint shader, GLsizei count, const GLchar* *string, const GLint *length);
    @mc
    def ShaderSource(self, GLuint shader, GLsizei count, char *string, GLint len):

        cdef char *s = &string[0]
        #printf('shader source: %s\n', s)
        _glShaderSource(shader, count, &s, &len) # &length[0])
        _glCompileShader(shader)

    @mc
    def CompileShader(self, GLuint shader): pass # _glCompileShader(shader)

    @mc
    def AttachShader(self, GLuint program, GLuint shader): _glAttachShader(program, shader)

    @mc
    def LinkProgram(self, GLuint program): _glLinkProgram(program)

    @mc
    def BindVertexArray(self, GLuint array): _glBindVertexArray(array)

    @mc
    def UseProgram(self, GLuint program): _glUseProgram(program)

    @mc
    def ValidateProgram( self, GLuint program): _glValidateProgram(program)


    #GLAPI void APIENTRY glUniform1i (GLint location, GLint v0);
    @mc
    def Uniform1i(self, GLint location, GLint v0):
        _glUniform1i(location, v0)

    #GLAPI void APIENTRY glUniform4f (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
    @mc
    def Uniform4f(self, GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3):
        _glUniform4f(location, v0, v1, v2, v3)

    @mc
    def Uniform1f(self, GLint location, GLfloat v0):
        _glUniform1f(location, v0)

    #GLAPI void APIENTRY glUniformMatrix4fv (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
    @mc
    def UniformMatrix4fv(self, GLint location, GLsizei count, GLboolean transpose, np.ndarray[float, ndim=2] value):
        _glUniformMatrix4fv(location, count, transpose, <float*>value.data)


    # GLAPI GLint APIENTRY glGetUniformLocation (GLuint program, const GLchar *name);
    @mc
    def GetUniformLocation(self, GLuint program, char *name):
        return _glGetUniformLocation(program, name)

    #def GetProgramInfoLog(	GLuint program, GLsizei maxLength, GLsizei *length, GLchar *infoLog):
    @mc
    def GetProgramInfoLog(self, GLuint program):
        cdef GLsizei maxLength
        maxLength = 1024
        cdef char infoLog[1024]
        cdef GLsizei returned_length
        returned_length = 0
        _glGetProgramInfoLog(program, maxLength, &returned_length, infoLog)
        return infoLog


    

