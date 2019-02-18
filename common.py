#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import numpy as np
from copy import deepcopy
import scipy.sparse as sp
from .cvwrap import cv2

try:
    from scipy.stats import nanmean as nanmean_impl
except:
    from numpy import nanmean as nanmean_impl

from chumpy.utils import row, col
from .contexts._constants import *

def nanmean(a, axis):
    # don't call nan_to_num in here, unless you check that
    # occlusion_test.py still works after you do it!
    result = nanmean_impl(a, axis=axis)
    return result

def nangradients(arr):
    dy = np.expand_dims(arr[:-1,:,:] - arr[1:,:,:], axis=3)
    dx = np.expand_dims(arr[:,:-1,:] - arr[:, 1:, :], axis=3)

    dy = np.concatenate((dy[1:,:,:], dy[:-1,:,:]), axis=3)
    dy = nanmean(dy, axis=3)
    dx = np.concatenate((dx[:,1:,:], dx[:,:-1,:]), axis=3)
    dx = nanmean(dx, axis=3)

    if arr.shape[2] > 1:
        gy, gx, _ = np.gradient(arr)
    else:
        gy, gx = np.gradient(arr.squeeze())
        gy = np.atleast_3d(gy)
        gx = np.atleast_3d(gx)
    gy[1:-1,:,:] = -dy
    gx[:,1:-1,:] = -dx

    return gy, gx



def dImage_wrt_2dVerts_bnd(observed, visible, visibility, barycentric, image_width, image_height, num_verts, f, bnd_bool):
    """Construct a sparse jacobian that relates 2D projected vertex positions
    (in the columns) to pixel values (in the rows). This can be done
    in two steps."""

    n_channels = np.atleast_3d(observed).shape[2]
    shape = visibility.shape

    # Step 1: get the structure ready, ie the IS and the JS
    IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
    JS = col(f[visibility.ravel()[visible]].ravel())
    JS = np.hstack((JS*2, JS*2+1)).ravel()

    pxs = np.asarray(visible % shape[1], np.int32)
    pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

    if n_channels > 1:
        IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
        JS = np.concatenate([JS for i in range(n_channels)])

    # Step 2: get the data ready, ie the actual values of the derivatives
    ksize = 1
    bndf = bnd_bool.astype(np.float64)
    nbndf = np.logical_not(bnd_bool).astype(np.float64)
    sobel_normalizer = cv2.Sobel(np.asarray(np.tile(row(np.arange(10)), (10, 1)), np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize)[5,5]

    bnd_nan = bndf.reshape((observed.shape[0], observed.shape[1], -1)).copy()
    bnd_nan.ravel()[bnd_nan.ravel()>0] = np.nan
    bnd_nan += 1
    obs_nonbnd = np.atleast_3d(observed) * bnd_nan

    ydiffnb, xdiffnb = nangradients(obs_nonbnd)

    observed = np.atleast_3d(observed)

    if observed.shape[2] > 1:
        ydiffbnd, xdiffbnd, _ = np.gradient(observed)
    else:
        ydiffbnd, xdiffbnd = np.gradient(observed.squeeze())
        ydiffbnd = np.atleast_3d(ydiffbnd)
        xdiffbnd = np.atleast_3d(xdiffbnd)

    # This corrects for a bias imposed boundary differences begin spread over two pixels
    # (by np.gradients or similar) but only counted once (since OpenGL's line
    # drawing spans 1 pixel)
    xdiffbnd *= 2.0
    ydiffbnd *= 2.0

    xdiffnb = -xdiffnb
    ydiffnb = -ydiffnb
    xdiffbnd = -xdiffbnd
    ydiffbnd = -ydiffbnd
    # ydiffnb *= 0
    # xdiffnb *= 0

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(121)
        plt.imshow(xdiffnb)
        plt.title('xdiffnb')
        plt.subplot(122)
        plt.imshow(xdiffbnd)
        plt.title('xdiffbnd')
        import pdb; pdb.set_trace()

    idxs = np.isnan(xdiffnb.ravel())
    xdiffnb.ravel()[idxs] = xdiffbnd.ravel()[idxs]

    idxs = np.isnan(ydiffnb.ravel())
    ydiffnb.ravel()[idxs] = ydiffbnd.ravel()[idxs]

    if True: # should be right thing
        xdiff = xdiffnb
        ydiff = ydiffnb
    else:  #should be old way
        xdiff = xdiffbnd
        ydiff = ydiffbnd


    # TODO: NORMALIZER IS WRONG HERE
    # xdiffnb = -cv2.Sobel(obs_nonbnd, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / np.atleast_3d(cv2.Sobel(row(np.arange(obs_nonbnd.shape[1])).astype(np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize))
    # ydiffnb = -cv2.Sobel(obs_nonbnd, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / np.atleast_3d(cv2.Sobel(col(np.arange(obs_nonbnd.shape[0])).astype(np.float64), cv2.CV_64F, dx=0, dy=1, ksize=ksize))
    #
    # xdiffnb.ravel()[np.isnan(xdiffnb.ravel())] = 0.
    # ydiffnb.ravel()[np.isnan(ydiffnb.ravel())] = 0.
    # xdiffnb.ravel()[np.isinf(xdiffnb.ravel())] = 0.
    # ydiffnb.ravel()[np.isinf(ydiffnb.ravel())] = 0.

    # xdiffnb = np.atleast_3d(xdiffnb)
    # ydiffnb = np.atleast_3d(ydiffnb)
    #
    # xdiffbnd = -cv2.Sobel(observed, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / sobel_normalizer
    # ydiffbnd = -cv2.Sobel(observed, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / sobel_normalizer
    #
    # xdiff = xdiffnb * np.atleast_3d(nbndf)
    # xdiff.ravel()[np.isnan(xdiff.ravel())] = 0
    # xdiff += xdiffbnd*np.atleast_3d(bndf)
    #
    # ydiff = ydiffnb * np.atleast_3d(nbndf)
    # ydiff.ravel()[np.isnan(ydiff.ravel())] = 0
    # ydiff += ydiffbnd*np.atleast_3d(bndf)

    #import pdb; pdb.set_trace()

    #xdiff = xdiffnb
    #ydiff = ydiffnb

    #import pdb; pdb.set_trace()

    datas = []

    # The data is weighted according to barycentric coordinates
    bc0 = col(barycentric[pys, pxs, 0])
    bc1 = col(barycentric[pys, pxs, 1])
    bc2 = col(barycentric[pys, pxs, 2])
    for k in range(n_channels):
        dxs = xdiff[pys, pxs, k]
        dys = ydiff[pys, pxs, k]
        if f.shape[1] == 3:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1,col(dxs)*bc2,col(dys)*bc2)).ravel())
        else:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())

    data = np.concatenate(datas)

    ij = np.vstack((IS.ravel(), JS.ravel()))
    result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

    return result



def dImage_wrt_2dVerts(observed, visible, visibility, barycentric, image_width, image_height, num_verts, f):
    """Construct a sparse jacobian that relates 2D projected vertex positions
    (in the columns) to pixel values (in the rows). This can be done
    in two steps."""

    n_channels = np.atleast_3d(observed).shape[2]
    shape = visibility.shape

    # Step 1: get the structure ready, ie the IS and the JS
    IS = np.tile(col(visible), (1, 2*f.shape[1])).ravel()
    JS = col(f[visibility.ravel()[visible]].ravel())
    JS = np.hstack((JS*2, JS*2+1)).ravel()

    pxs = np.asarray(visible % shape[1], np.int32)
    pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

    if n_channels > 1:
        IS = np.concatenate([IS*n_channels+i for i in range(n_channels)])
        JS = np.concatenate([JS for i in range(n_channels)])

    # Step 2: get the data ready, ie the actual values of the derivatives
    ksize=1
    sobel_normalizer = cv2.Sobel(np.asarray(np.tile(row(np.arange(10)), (10, 1)), np.float64), cv2.CV_64F, dx=1, dy=0, ksize=ksize)[5,5]
    xdiff = -cv2.Sobel(observed, cv2.CV_64F, dx=1, dy=0, ksize=ksize) / sobel_normalizer
    ydiff = -cv2.Sobel(observed, cv2.CV_64F, dx=0, dy=1, ksize=ksize) / sobel_normalizer

    xdiff = np.atleast_3d(xdiff)
    ydiff = np.atleast_3d(ydiff)

    datas = []

    # The data is weighted according to barycentric coordinates
    bc0 = col(barycentric[pys, pxs, 0])
    bc1 = col(barycentric[pys, pxs, 1])
    bc2 = col(barycentric[pys, pxs, 2])
    for k in range(n_channels):
        dxs = xdiff[pys, pxs, k]
        dys = ydiff[pys, pxs, k]
        if f.shape[1] == 3:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1,col(dxs)*bc2,col(dys)*bc2)).ravel())
        else:
            datas.append(np.hstack((col(dxs)*bc0,col(dys)*bc0,col(dxs)*bc1,col(dys)*bc1)).ravel())

    data = np.concatenate(datas)

    ij = np.vstack((IS.ravel(), JS.ravel()))
    result = sp.csc_matrix((data, ij), shape=(image_width*image_height*n_channels, num_verts*2))

    return result

def flow_to(self, v_next, cam_next):
    from chumpy.ch import MatVecMult

    color_image = self.r
    visibility = self.visibility_image
    pxpos = np.zeros_like(self.color_image)
    pxpos[:,:,0] = np.tile(row(np.arange(self.color_image.shape[1])), (self.color_image.shape[0], 1))
    pxpos[:,:,2] = np.tile(col(np.arange(self.color_image.shape[0])), (1, self.color_image.shape[1]))

    visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    num_visible = len(visible)

    barycentric = self.barycentric_image


    # map 3d to 3d
    JS = col(self.f[visibility.ravel()[visible]]).ravel()
    IS = np.tile(col(np.arange(JS.size/3)), (1, 3)).ravel()
    data = barycentric.reshape((-1,3))[visible].ravel()

    # replicate to xyz
    IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
    JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
    data = np.concatenate((data, data, data))

    verts_to_visible = sp.csc_matrix((data, (IS, JS)), shape=(np.max(IS)+1, self.v.r.size))

    v_old = self.camera.v
    cam_old = self.camera

    if cam_next is None:
        cam_next = self.camera

    self.camera.v = MatVecMult(verts_to_visible, self.v.r)
    r1 = self.camera.r.copy()

    self.camera = cam_next
    self.camera.v = MatVecMult(verts_to_visible, v_next)
    r2 = self.camera.r.copy()

    n_channels = self.camera.shape[1]
    flow = r2 - r1
    flow_im = np.zeros((self.frustum['height'], self.frustum['width'], n_channels)).reshape((-1,n_channels))

    flow_im[visible] = flow
    flow_im = flow_im.reshape((self.frustum['height'], self.frustum['width'], n_channels))

    self.camera = cam_old
    self.camera.v = v_old
    return flow_im


def dr_wrt_bgcolor(visibility, frustum, num_channels):
    invisible = np.nonzero(visibility.ravel() == 4294967295)[0]
    IS = invisible
    JS = np.zeros(len(IS))
    data = np.ones(len(IS))

    # color image, so 3 channels
    IS = np.concatenate([IS*num_channels+k for k in range(num_channels)])
    JS = np.concatenate([JS*num_channels+k for k in range(num_channels)])
    data = np.concatenate([data for i in range(num_channels)])
    # IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
    # JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
    # data = np.concatenate((data, data, data))

    ij = np.vstack((IS.ravel(), JS.ravel()))
    result = sp.csc_matrix((data, ij), shape=(frustum['width']*frustum['height']*num_channels, num_channels))
    return result


def dr_wrt_vc(visible, visibility, f, barycentric, frustum, vc_size, num_channels):
    # Each pixel relies on three verts
    IS = np.tile(col(visible), (1, 3)).ravel()
    JS = col(f[visibility.ravel()[visible]].ravel())

    bc = barycentric.reshape((-1,3))
    data = np.asarray(bc[visible,:], order='C').ravel()

    IS = np.concatenate([IS*num_channels+k for k in range(num_channels)])
    JS = np.concatenate([JS*num_channels+k for k in range(num_channels)])
    data = np.concatenate([data for i in range(num_channels)])
    # IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
    # JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
    # data = np.concatenate((data, data, data))

    ij = np.vstack((IS.ravel(), JS.ravel()))
    result = sp.csc_matrix((data, ij), shape=(frustum['width']*frustum['height']*num_channels, vc_size))
    return result


def draw_visibility_image(gl, v, f, boundarybool_image=None):
    v = np.asarray(v)
    gl.Disable(GL_TEXTURE_2D)
    gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)

    result = draw_visibility_image_internal(gl, v, f)
    if boundarybool_image is None:
        return result

    rr = result.ravel()
    faces_to_draw = np.unique(rr[rr != 4294967295])
    if len(faces_to_draw)==0:
        result = np.ones((gl.height, gl.width)).astype(np.uint32)*4294967295
        return result
    gl.PolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    result2 = draw_visibility_image_internal(gl, v, f[faces_to_draw])
    gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    bbi = boundarybool_image

    result2 = result2.ravel()
    idxs = result2 != 4294967295
    result2[idxs] = faces_to_draw[result2[idxs]]

    if False:
        result2[result2==4294967295] = 0
        import matplotlib.pyplot as plt
        result2 = result2.reshape(result.shape[:2])
        plt.figure()
        plt.subplot(121)
        plt.imshow(result.squeeze())
        plt.subplot(122)
        plt.imshow(result2.squeeze())

    result2 = result2.reshape(result.shape[:2])
    result = result2 * bbi + result * (1 - bbi)
    return result



def draw_visibility_image_internal(gl, v, f):
    """Assumes camera is set up correctly in gl context."""
    gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    fc = np.arange(1, len(f)+1)
    fc = np.tile(col(fc), (1, 3))
    fc[:, 0] = fc[:, 0] & 255
    fc[:, 1] = (fc[:, 1] >> 8 ) & 255
    fc[:, 2] = (fc[:, 2] >> 16 ) & 255
    fc = np.asarray(fc, dtype=np.uint8)

    draw_colored_primitives(gl, v, f, fc)
    raw = np.asarray(gl.getImage(), np.uint32)
    raw = raw[:,:,0] + raw[:,:,1]*256 + raw[:,:,2]*256*256 - 1
    return raw

# this assumes that fc is either "by faces" or "verts by face", not "by verts"
def draw_colored_primitives(gl, v, f, fc=None):

    gl.EnableClientState(GL_VERTEX_ARRAY);
    verts_by_face = np.asarray(v.reshape((-1,3))[f.ravel()], dtype=np.float64, order='C')
    gl.VertexPointer(verts_by_face)

    if fc is not None:
        gl.EnableClientState(GL_COLOR_ARRAY);
        if fc.size == verts_by_face.size:
            vc_by_face = fc
        else:
            vc_by_face = np.repeat(fc, f.shape[1], axis=0)

        if vc_by_face.size != verts_by_face.size:
            raise Exception('fc must have either rows=(#rows in faces) or rows=(# elements in faces)')

        if isinstance(fc[0,0], np.float64):
            vc_by_face = np.asarray(vc_by_face, dtype=np.float64, order='C')
            gl.ColorPointerd(vc_by_face)
        elif isinstance(fc[0,0], np.uint8):
            vc_by_face = np.asarray(vc_by_face, dtype=np.uint8, order='C')
            gl.ColorPointerub(vc_by_face)
        else:
            raise Exception('Unknown color type for fc')
    else:
        gl.DisableClientState(GL_COLOR_ARRAY);


    if f.shape[1]==2:
        primtype = GL_LINES
    else:
        primtype = GL_TRIANGLES
    gl.DrawElements(primtype, np.arange(f.size, dtype=np.uint32).ravel())

    if primtype == GL_LINES:
        f = np.fliplr(f).copy()
        verts_by_edge = v.reshape((-1,3))[f.ravel()]
        verts_by_edge = np.asarray(verts_by_edge, dtype=np.float64, order='C')
        gl.VertexPointer(verts_by_edge)
        gl.DrawElements(GL_LINES, np.arange(f.size, dtype=np.uint32).ravel())


def draw_texcoord_image(glf, v, f, vt, ft, boundarybool_image=None):
    gl = glf
    gl.Disable(GL_TEXTURE_2D)
    gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)

    gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    # want vtc: texture-coordinates per vertex (not per element in vc)
    colors = vt[ft.ravel()]

    colors = np.asarray(np.hstack((colors, col(colors[:,0]*0))), np.float64, order='C')
    draw_colored_primitives(gl, v, f, colors)

    if boundarybool_image is not None:
        gl.PolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        draw_colored_primitives(gl, v, f, colors)
        gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    result = np.asarray(deepcopy(gl.getImage()), np.float64, order='C')[:,:,:2].copy()
    result[:,:,1] = 1. - result[:,:,1]
    return result


def draw_barycentric_image(gl, v, f, boundarybool_image=None):
    v = np.asarray(v)
    without_overdraw = draw_barycentric_image_internal(gl, v, f)
    if boundarybool_image is None:
        return without_overdraw

    gl.PolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    overdraw = draw_barycentric_image_internal(gl, v, f)
    gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    bbi = np.atleast_3d(boundarybool_image)
    return bbi * overdraw + (1. - bbi) * without_overdraw


def draw_barycentric_image_internal(gl, v, f):

    gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    gl.EnableClientState(GL_VERTEX_ARRAY);
    gl.EnableClientState(GL_COLOR_ARRAY);

    verts_by_face = v.reshape((-1,3))[f.ravel()]
    verts_by_face = np.asarray(verts_by_face, dtype=np.float64, order='C')
    vc_by_face = np.asarray(np.tile(np.eye(3)[:f.shape[1], :], (verts_by_face.shape[0]/f.shape[1], 1)), order='C')

    gl.ColorPointerd(vc_by_face)
    gl.VertexPointer(verts_by_face)
    gl.DrawElements(GL_TRIANGLES if f.shape[1]==3 else GL_LINES, np.arange(f.size, dtype=np.uint32).ravel())
    result = np.asarray(deepcopy(gl.getImage()), np.float64)

    return result


# May end up using this, maybe not
def get_inbetween_boundaries(self):
    camera = self.camera
    frustum = self.frustum
    w = frustum['width']
    h = frustum['height']
    far = frustum['far']
    near = frustum['near']

    self.glb.Viewport(0, 0, w-1, h)
    _setup_camera(self.glb,
                  camera.c.r[0]-.5, camera.c.r[1],
                  camera.f.r[0], camera.f.r[1],
                  w-1, h,
                  near, far,
                  camera.view_matrix, camera.k)
    bnd_x = draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.camera)[:,:-1]

    self.glb.Viewport(0, 0, w, h-1)
    _setup_camera(self.glb,
                  camera.c.r[0], camera.c.r[1]-.5,
                  camera.f.r[0], camera.f.r[1],
                  w, h-1,
                  near, far,
                  camera.view_matrix, camera.k)
    bnd_y = draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.camera)[:-1,:]

    # Put things back to normal
    self.glb.Viewport(0, 0, w, h)
    setup_camera(self.glb, camera, frustum)
    return bnd_x, bnd_y

