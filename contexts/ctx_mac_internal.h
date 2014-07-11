/*
Author(s): Matthew Loper

See LICENCE.txt for licensing information.
*/

#include "gl_includes.h"

void *create_context (unsigned int imageWidth, unsigned int imageHeight, GLenum typ); // =GL_UNSIGNED_BYTE
void set_current(void *ctx);
void release_context(void *ctx);

