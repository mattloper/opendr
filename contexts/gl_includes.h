/*
Author(s): Matthew Loper

See LICENCE.txt for licensing information.
*/

#define GL_GLEXT_PROTOTYPES 1

#ifdef __OSMESA__
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#else
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/CGLTypes.h>
#endif
#endif

