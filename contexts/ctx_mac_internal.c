/*
Author(s): Matthew Loper

See LICENCE.txt for licensing information.
*/

#include <stdio.h>
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <OpenGL/OpenGL.h>
#include <OpenGL/CGLTypes.h>


void release_context(void *ctx)
{
    CGLDestroyContext((CGLContextObj)ctx);
}

void set_current(void *ctx)
{
    CGLSetCurrentContext((CGLContextObj)ctx);
}


void *create_context (unsigned int imageWidth, unsigned int imageHeight, GLenum typ)
{
    
    
    // These identifiers are described here:
    // https://developer.apple.com/library/mac/#documentation/graphicsimaging/reference/CGL_OpenGL/Reference/reference.html
    CGLPixelFormatAttribute float_attribs[] =
    {
        kCGLPFAAccelerated,
        kCGLPFAColorSize, (CGLPixelFormatAttribute)(32*4), // 24,
        kCGLPFADepthSize, (CGLPixelFormatAttribute)16,
        kCGLPFAColorFloat,
	    //kCGLPFAOpenGLProfile, kCGLOGLPVersion_3_2_Core,
        (CGLPixelFormatAttribute)0,
    };

    CGLPixelFormatAttribute ubyte_attribs[] =
    {
        kCGLPFAAccelerated,
        kCGLPFAColorSize, (CGLPixelFormatAttribute)24, // 24,
        kCGLPFADepthSize, (CGLPixelFormatAttribute)16,
	    //kCGLPFAOpenGLProfile, kCGLOGLPVersion_3_2_Core,
        (CGLPixelFormatAttribute)0,
    };

    CGLPixelFormatAttribute *attribs;
    if (typ == GL_UNSIGNED_BYTE)
        attribs = ubyte_attribs;
    else if (typ == GL_FLOAT)
        attribs = float_attribs;
    else {
        printf("%s: %s(): typ parameter must be GL_UNSIGNED_BYTE or GL_FLOAT\n", __FILE__, __FUNCTION__);
        exit(0);
        return NULL;
    }

    
    CGLPixelFormatObj pixelFormatObj;
    GLint numPixelFormats;
    
    CGLChoosePixelFormat (attribs, &pixelFormatObj, &numPixelFormats);
    
    if( pixelFormatObj == NULL ) {
        // Your code to notify the user and take action.
        printf("CGLChoosePixelFormat failure!\n");
        return NULL;
    }
    
    // Create context
    CGLContextObj cglContext1 = NULL;
    CGLCreateContext(pixelFormatObj, NULL, &cglContext1);
    CGLSetCurrentContext(cglContext1);
    
    // Allocate frame and renderbuffer
    GLuint framebuffer, renderbuffer;
    glGenFramebuffersEXT(1, &framebuffer);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuffer);
    glGenRenderbuffersEXT(1, &renderbuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderbuffer);
    
    
    //glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA32F_ARB, imageWidth, imageHeight);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA8, imageWidth, imageHeight);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                                 GL_RENDERBUFFER_EXT, renderbuffer);
    GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
    
    
    // Get depth going
    GLuint depthRenderbuffer;
    glGenRenderbuffersEXT(1, &depthRenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, imageWidth, imageHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
    
	// "The system retains the pixel format object when you call the function CGLCreateContext, 
	// so you can release a pixel format object immediately after passing it to the context creation function."
    CGLReleasePixelFormat(pixelFormatObj); 
    if (status != GL_FRAMEBUFFER_COMPLETE_EXT)
    {
        printf("uh oh, bad status from glCheckFramebufferStatusEXT!\n");
        return NULL;
    }

    //fprintf(stderr, "VENDOR: %s\n", (char *)glGetString(GL_VENDOR));
    //fprintf(stderr, "VERSION: %s\n", (char *)glGetString(GL_VERSION));
    //fprintf(stderr, "RENDERER: %s\n", (char *)glGetString(GL_RENDERER));

    return cglContext1;
}

