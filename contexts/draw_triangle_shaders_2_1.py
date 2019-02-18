__author__ = 'matt'

import cv2
import numpy as np
from opendr.contexts._constants import *

import platform
if platform.system()=='Darwin':
    from ctx_mac import OsContext
else:
    from ctx_mesa import OsContext

fs_source = """
#version 120

void main(void) {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

vs_source = """
#version 120
void main(void) {
    //gl_Position = ftransform();
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
}
"""

def main():
    w = 640
    h = 480
    gl = OsContext(w, h, typ=GL_FLOAT)
    gl.Viewport(0, 0, w, h)

    gl.MatrixMode(GL_PROJECTION)
    gl.LoadIdentity();

    gl.MatrixMode(GL_MODELVIEW);
    gl.LoadIdentity()

    gl.Enable(GL_DEPTH_TEST)
    gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    gl.Disable(GL_LIGHTING);
    gl.Disable(GL_CULL_FACE)
    gl.PixelStorei(GL_PACK_ALIGNMENT,1)
    gl.PixelStorei(GL_UNPACK_ALIGNMENT,1)

    gl.Clear(GL_COLOR_BUFFER_BIT)
    gl.Clear(GL_DEPTH_BUFFER_BIT)

    v = np.random.rand(9).reshape((-1,3))
    f = np.arange(v.size, dtype=np.uint32)
    gl.EnableClientState(GL_VERTEX_ARRAY)
    gl.VertexPointer(v)

    if True:
        program = gl.CreateProgram()
        fs = gl.CreateShader(GL_FRAGMENT_SHADER)
        gl.ShaderSource(fs, 1, fs_source, len(fs_source))
        vs = gl.CreateShader(GL_VERTEX_SHADER)
        gl.ShaderSource(vs, 1, vs_source, len(vs_source))
        gl.AttachShader(program, vs)
        gl.AttachShader(program, fs)
        gl.LinkProgram(program)
        gl.UseProgram(program)

        print('glValidateProgram: ' + str(gl.ValidateProgram(program)))
        print('glGetProgramInfoLog ' + str(gl.GetProgramInfoLog(program)))
        print('GL_MAX_VERTEX_ATTRIBS: ' + str(gl.GetInteger(GL_MAX_VERTEX_ATTRIBS)))


    gl.DrawElements(GL_TRIANGLES, f)

    im = gl.getImage()
    cv2.imshow('a', im)
    import pdb; pdb.set_trace()






if __name__ == '__main__':
    main()