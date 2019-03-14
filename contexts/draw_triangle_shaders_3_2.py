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
#version 150

in vec3 Color;
out vec4 outputF;

void main()
{
    //outputF = vec4(Color,1.0);
    outputF = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

# fs_source = """
# #version 150
#
#
# void main()
# {
# }
# """

vs_source = """
#version 150

uniform mat4 viewMatrix, projMatrix;

in vec4 position;
in vec3 color;

out vec3 Color;

void main()
{
    Color = color;
    gl_Position = projMatrix * viewMatrix * position ;
    //gl_Position = vec4(.5, .6, .7, 1.0);
}
"""

def main():
    w = 640
    h = 480
    gl = OsContext(w, h, typ=GL_FLOAT)
    gl.Viewport(0, 0, w, h)

    gl.Enable(GL_DEPTH_TEST)
    gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    # gl.Disable(GL_LIGHTING); # causes error
    gl.Disable(GL_CULL_FACE)
    gl.PixelStorei(GL_PACK_ALIGNMENT,1)
    gl.PixelStorei(GL_UNPACK_ALIGNMENT,1)

    gl.Clear(GL_COLOR_BUFFER_BIT)
    gl.Clear(GL_DEPTH_BUFFER_BIT)

    vertices = np.random.rand(9).reshape((-1,3)).ravel()
    #vertices = np.ones(9) * .5
    #vertices = np.arange(9).astype(np.float64)
    colors = np.ones_like(vertices).ravel()
    f = np.arange(vertices.size, dtype=np.uint32)
    #gl.EnableClientState(GL_VERTEX_ARRAY)
    #gl.VertexPointer(v)


    ############################
    # ENABLE SHADER
    program = gl.CreateProgram()
    fs = gl.CreateShader(GL_FRAGMENT_SHADER)
    gl.ShaderSource(fs, 1, fs_source, len(fs_source))
    vs = gl.CreateShader(GL_VERTEX_SHADER)
    gl.ShaderSource(vs, 1, vs_source, len(vs_source))
    gl.AttachShader(program, vs)
    gl.AttachShader(program, fs)

    gl.BindFragDataLocation(program, 0, "outputF")

    gl.LinkProgram(program)

    vertexLoc = gl.GetAttribLocation(program,"position")
    colorLoc = gl.GetAttribLocation(program,"color")
    projMatrixLoc = gl.GetUniformLocation(program, "projMatrix");
    viewMatrixLoc = gl.GetUniformLocation(program, "viewMatrix");

    gl.UseProgram(program)

    #############################
    # CREATE VERTEX ARRAY OBJECT
    vao = gl.GenVertexArrays(3)
    gl.BindVertexArray(vao[0])
    buffers = gl.GenBuffers(3)

    # VERTICES
    gl.BindBuffer(GL_ARRAY_BUFFER, buffers[0])
    gl.BufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
    gl.VertexAttribPointer(vertexLoc, 3, GL_DOUBLE, 0, 0)
    gl.EnableVertexAttribArray(vertexLoc)

    # COLORS
    gl.BindBuffer(GL_ARRAY_BUFFER, buffers[1])
    gl.BufferData(GL_ARRAY_BUFFER, colors, GL_STATIC_DRAW)
    gl.VertexAttribPointer(colorLoc, 3, GL_DOUBLE, 0, 0)
    gl.EnableVertexAttribArray(colorLoc)

    # TRIANGULATION
    gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2])
    gl.BufferData(GL_ELEMENT_ARRAY_BUFFER, f, GL_STATIC_DRAW)


    # PROJECTION MATRICES
    projMatrix = np.eye(4, dtype=np.float32)
    viewMatrix = np.eye(4, dtype=np.float32)
    gl.UniformMatrix4fv(projMatrixLoc,  1, 0, projMatrix)
    gl.UniformMatrix4fv(viewMatrixLoc,  1, 0, viewMatrix)


    ###############################
    # DRAWING
    gl.BindVertexArray(vao[0])
    gl.DrawElements2(GL_TRIANGLES, f.size, GL_UNSIGNED_INT)

    print('glValidateProgram: ' + str(gl.ValidateProgram(program)))
    print('glGetProgramInfoLog ' + str(gl.GetProgramInfoLog(program)))
    print('GL_MAX_VERTEX_ATTRIBS: ' + str(gl.GetInteger(GL_MAX_VERTEX_ATTRIBS)))

    im = gl.getImage()
    cv2.imshow('a', im)
    print(gl.GetError())
    import pdb; pdb.set_trace()






if __name__ == '__main__':
    main()