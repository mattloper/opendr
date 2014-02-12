#!/usr/bin/env python
# encoding: utf-8

"""
Copyright (C) 2013
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['ColoredRenderer', 'TexturedRenderer', 'DepthRenderer']

import numpy as np
import cv2
import time
import platform
import scipy.sparse as sp
from copy import deepcopy
from . import common
from .common import draw_visibility_image, draw_barycentric_image, draw_colored_primitives
from .topology import get_vertices_per_edge, get_faces_per_edge

if platform.system()=='Darwin':
    from .contexts.ctx_mac import OsContext
else:
    from .contexts.ctx_mesa import OsContext

from chumpy import *
from .contexts._constants import *
from chumpy.utils import row, col


pixel_center_offset = 0.5





class DepthRenderer(Ch):
    terms = 'f', 'frustum', 'background_image'
    dterms = 'camera', 'v'        
    
    @property
    def v(self):
        return self.camera.v
    
    @v.setter
    def v(self, newval):
        self.camera.v = newval
    
    def compute_r(self):
        return self.depth_image.reshape((self.frustum['height'], self.frustum['width']))
        
    def compute_dr_wrt(self, wrt):
        
        if wrt is not self.camera and wrt is not self.v:
            return None
        
        visibility = self.visibility_image
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        if wrt is self.camera:
            
            shape = visibility.shape        
            depth = self.depth_image
            
            barycentric = self.barycentric_image
    
            result1 = common.dImage_wrt_2dVerts(depth, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)

            return result1

        elif wrt is self.v:
            return None
            IS = np.tile(col(visible), (1, 9)).ravel()
            JS = col(self.f[visibility.ravel()[visible]].ravel())
            JS = np.hstack((JS*3, JS*3+1, JS*3+2)).ravel()
            rays = self.glb.getDepthCloud(1.5*np.ones((self.frustum['height'],self.frustum['width']))) - self.glb.getDepthCloud(0.5*np.ones((self.frustum['height'],self.frustum['width'])))
            data = np.tile(rays[visible]/3., (1,3)).ravel()
            result2 = sp.csc_matrix((data, (IS, JS)), shape=(self.frustum['height']*self.frustum['width'], self.v.r.size))
            return -result2
            
    
    def on_changed(self, which):

        if 'frustum' in which:
            w = self.frustum['width']
            h = self.frustum['height']
            self.glf = OsContext(w, h, typ=GL_FLOAT)
            self.glf.Viewport(0, 0, w, h)
            self.glb = OsContext(w, h, typ=GL_UNSIGNED_BYTE)
            self.glb.Viewport(0, 0, w, h)
            
            
        if 'frustum' in which or 'camera' in which:
            setup_camera(self.glb, self.camera, self.frustum)
            setup_camera(self.glf, self.camera, self.frustum)
            
        assert(self.v is self.camera.v)
            
        #if ('v' in which and hasattr(self, 'camera')) or ('camera' in which and hasattr(self, 'v')):
        #    self.camera.v = self.v
            
        #if ('v' in which and hasattr(self, 'camera')): 
        #    self.camera.v = self.v
        #else:
        #    assert(self.v is self.camera.v)

    @depends_on('f', 'camera', 'background_image')
    def depth_image(self):
        self._call_on_changed()

        gl = self.glb
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # use face colors if given
        draw_noncolored_verts(gl, self.camera.v.r, self.f)

        result = np.asarray(deepcopy(gl.getDepth()), np.float64)

        if hasattr(self, 'background_image'):            
            visibility = self.visibility_image
            visible = np.nonzero(visibility.ravel() != 4294967295)[0]
            result2 = np.require(self.background_image.copy(), dtype=np.float64)
            result2.ravel()[visible] = result.ravel()[visible]
            result = result2
        
        return result
        

    @depends_on('f', 'camera')
    def barycentric_image(self):
        self._call_on_changed()
        return draw_barycentric_image(self.glf, self.camera.v.r, self.f)
    
    @depends_on('f', 'camera')
    def visibility_image(self):
        self._call_on_changed()
        return draw_visibility_image(self.glb, self.camera.v.r, self.f)
    
    def getDepthMesh(self, depth_image=None):
        self._call_on_changed() # make everything is up-to-date
        v = self.glb.getDepthCloud(depth_image)
        w = self.frustum['width']
        h = self.frustum['height']
        idxs = np.arange(w*h).reshape((h, w))
    
        # v0 is upperleft, v1 is upper right, v2 is lowerleft, v3 is lowerright
        v0 = col(idxs[:-1,:-1])
        v1 = col(idxs[:-1,1:])
        v2 = col(idxs[1:,:-1])
        v3 = col(idxs[1:,1:])

        f = np.hstack((v0, v1, v2, v1, v3, v2)).reshape((-1,3))        
        return v, f


class BoundaryRenderer(Ch):
    terms = 'f', 'frustum', 'num_channels'
    dterms = 'camera',

    @property
    def v(self):
        return self.camera.v

    @v.setter
    def v(self, newval):
        self.camera.v = newval

    def compute_r(self):
        return self.color_image
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.camera:
            return None
        
        visibility = self.boundaryid_image
        shape = visibility.shape        
        
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        num_visible = len(visible)

        barycentric = self.barycentric_image
    
        return common.dImage_wrt_2dVerts(self.color_image, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.vpe)
        

    def on_changed(self, which):
        if 'frustum' in which:
            w = self.frustum['width']
            h = self.frustum['height']
            self.glf = OsContext(w, h, typ=GL_FLOAT)
            self.glf.Viewport(0, 0, w, h)
            self.glb = OsContext(w, h, typ=GL_UNSIGNED_BYTE)
            self.glb.Viewport(0, 0, w, h)
            
        if 'frustum' in which or 'camera' in which:
            setup_camera(self.glb, self.camera, self.frustum)
            setup_camera(self.glf, self.camera, self.frustum)
            
        # if 'v' in which and hasattr(self, 'camera') or 'camera' in which and hasattr(self, 'v'):
        #     self.camera.v = self.v


    @depends_on('f', 'frustum', 'camera', 'v')
    def boundarybool_image(self):
        self._call_on_changed()

        boundaryid_image = self.boundaryid_image
        return np.asarray(boundaryid_image != 4294967295, np.float64).reshape(boundaryid_image.shape)


    @depends_on(terms+dterms)
    def boundaryid_image(self):
        self._call_on_changed()
        return draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.camera)

    @depends_on('f', 'camera', 'v')
    def barycentric_image(self):
        self._call_on_changed()
        return draw_barycentric_image(self.glf, self.v.r, self.vpe)
    
    @depends_on('f')
    def primitives_per_edge(self):
        v=self.v.r.reshape((-1,3))
        f=self.f
        fpe = get_faces_per_edge(v, f)
        vpe = get_vertices_per_edge(v, f, fpe)
        return fpe, vpe

    @property
    def vpe(self):
        return self.primitives_per_edge[1]
    
    @property
    def fpe(self):
        return self.primitives_per_edge[0]
    
    @depends_on(terms+dterms)
    def color_image(self):
        self._call_on_changed()
        result = self.boundarybool_image
        return np.dstack([result for i in range(self.num_channels)])



class ColoredRenderer(Ch):
    terms = 'f', 'frustum', 'background_image', 'overdraw'
    dterms = 'vc', 'camera', 'bgcolor'        
    
    
    @property
    def v(self):
        return self.camera.v
    
    @v.setter
    def v(self, newval):
        self.camera.v = newval
    
    def compute_r(self):
        return self.color_image.reshape((self.frustum['height'], self.frustum['width'], -1))
        
    def compute_dr_wrt(self, wrt):
        if wrt is not self.camera and wrt is not self.vc and wrt is not self.bgcolor:
            return None
        
        visibility = self.visibility_image

        shape = visibility.shape
        color = self.color_image
        
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        num_visible = len(visible)

        barycentric = self.barycentric_image

        if wrt is self.camera:
            if self.overdraw:
                return common.dImage_wrt_2dVerts_bnd(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f, self.boundaryid_image != 4294967295)
            else:
                return common.dImage_wrt_2dVerts(color, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)

        elif wrt is self.vc:
            return common.dr_wrt_vc(visible, visibility, self.f, barycentric, self.frustum, self.v.size)

        elif wrt is self.bgcolor:
            return common.dr_wrt_bgcolor(visibility, self.frustum)


    def on_changed(self, which):
        if 'frustum' in which:
            w = self.frustum['width']
            h = self.frustum['height']
            self.glf = OsContext(w, h, typ=GL_FLOAT)
            self.glf.Viewport(0, 0, w, h)
            self.glb = OsContext(w, h, typ=GL_UNSIGNED_BYTE)
            self.glb.Viewport(0, 0, w, h)
            
        if 'frustum' in which or 'camera' in which:
            setup_camera(self.glb, self.camera, self.frustum)
            setup_camera(self.glf, self.camera, self.frustum)
            
        if not hasattr(self, 'bgcolor'):
            self.bgcolor = Ch(np.array([.5,.5,.5]))
            which.add('bgcolor')

        if not hasattr(self, 'overdraw'):
            self.overdraw = True
            
        if 'bgcolor' in which or ('frustum' in which and hasattr(self, 'bgcolor')):
            self.glf.ClearColor(self.bgcolor.r[0], self.bgcolor.r[1], self.bgcolor.r[2], 1.)
            

    def flow_to(self, v_next, cam_next=None):
        return common.flow_to(self, v_next, cam_next)



    def filter_for_triangles(self, which_triangles):
        cim = self.color_image
        vim = self.visibility_image+1
        arr = np.zeros(len(self.f)+1)
        arr[which_triangles+1] = 1
        
        relevant_pixels = arr[vim.ravel()]
        cim2 = cim.copy() * np.atleast_3d(relevant_pixels.reshape(vim.shape))
        relevant_pixels = np.nonzero(arr[vim.ravel()])[0]
        xs = relevant_pixels % vim.shape[1]
        ys = relevant_pixels / vim.shape[1]
        return cim2[np.min(ys):np.max(ys), np.min(xs):np.max(xs), :]
        

    @depends_on('f')
    def primitives_per_edge(self):
        v=self.v.r.reshape((-1,3))
        f=self.f
        fpe = get_faces_per_edge(v, f)
        vpe = get_vertices_per_edge(v, f, fpe)
        return fpe, vpe

    @property
    def vpe(self):
        return self.primitives_per_edge[1]
    
    @property
    def fpe(self):
        return self.primitives_per_edge[0]
    


    @depends_on('f', 'camera', 'vc')
    def boundarycolor_image(self):

        try:
            return self.draw_boundarycolor_image(with_vertex_colors=True)
        except:
            import pdb; pdb.set_trace()


    def draw_color_image(self, gl):
        self._call_on_changed()
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if hasattr(self, 'background_image'):
            gl.MatrixMode(GL_PROJECTION)
            gl.PushMatrix()
            gl.LoadIdentity()
            gl.Ortho(0, gl.width, 0, gl.height)
            gl.MatrixMode(GL_MODELVIEW)
            gl.PushMatrix()
            gl.LoadIdentity()
            gl.RasterPos2i(0, 0)

            gl.DrawPixels(self.background_image.shape[1],self.background_image.shape[0],GL_RGB,GL_FLOAT,np.asarray(self.background_image, np.float32))
            gl.PopMatrix() # modelview
            gl.MatrixMode(GL_PROJECTION)
            gl.PopMatrix() # projection
            assert (not gl.GetError())
            gl.Clear(GL_DEPTH_BUFFER_BIT)

        # use face colors if given
        draw_colored_verts(gl, self.v.r, self.f, self.vc.r)

        return np.asarray(deepcopy(gl.getImage()), np.float64)

    @depends_on(dterms+terms)
    def color_image(self):
        gl = self.glf
        gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        no_overdraw = self.draw_color_image(gl)

        if not self.overdraw:
            return no_overdraw

        gl.PolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        overdraw = self.draw_color_image(gl)
        gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        return overdraw*self.boundarybool_image + no_overdraw*(1-self.boundarybool_image)



    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def barycentric_image(self):
        self._call_on_changed()
        without_overdraw = draw_barycentric_image(self.glf, self.v.r, self.f)
        if not self.overdraw:
            return without_overdraw

        self.glf.PolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        overdraw = draw_barycentric_image(self.glf, self.v.r, self.f)
        self.glf.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        bbi = self.boundarybool_image
        return bbi * overdraw + (1. - bbi) * without_overdraw

    @depends_on('f', 'frustum', 'camera', 'overdraw')
    def visibility_image(self):
        self._call_on_changed()

        self.glb.Disable(GL_TEXTURE_2D)
        self.glb.DisableClientState(GL_TEXTURE_COORD_ARRAY)

        if not self.overdraw:
            return draw_visibility_image(self.glb, self.v.r, self.f)

        result = np.atleast_3d(draw_visibility_image(self.glb, self.v.r, self.f))
        rr = result.ravel()
        faces_to_draw = np.unique(rr[rr != 4294967295])
        self.glb.PolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        result2 = np.atleast_3d(draw_visibility_image(self.glb, self.v.r, self.f[faces_to_draw]))
        self.glb.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        bbi = self.boundarybool_image

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

        result2 = np.atleast_3d(result2.reshape(result.shape[:2]))
        result = result2 * bbi + result * (1 - bbi)

        return result


    @depends_on(dterms+terms)
    def boundaryid_image(self):
        self._call_on_changed()
        return draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.camera)

    @property
    def boundarybool_image(self):
        return np.atleast_3d(self.boundaryid_image != 4294967295)

    @depends_on('f', 'frustum', 'camera')
    def boundary_images(self):
        self._call_on_changed()
        return draw_boundary_images(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.camera)

    @depends_on(terms+dterms)    
    def boundarycolor_image(self): 
        self._call_on_changed()
        try:
            gl = self.glf
            colors = self.vc.r.reshape((-1,3))[self.vpe.ravel()]
            gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            draw_colored_primitives(gl, self.v.r.reshape((-1,3)), self.vpe, colors)
            return np.asarray(deepcopy(gl.getImage()), np.float64)
        except:
            import pdb; pdb.set_trace()
            


        
class TexturedRenderer(ColoredRenderer):
    terms = 'f', 'frustum', 'texture_image', 'vt', 'ft', 'background_image', 'overdraw'
    dterms = 'vc', 'camera', 'bgcolor'

    def __del__(self):
        self.release_textures()
        
    def release_textures(self):
        if hasattr(self, 'textureID'):
            arr = np.asarray(np.array([self.textureID]), np.uint32)
            self.glf.DeleteTextures(arr)

    @property
    def v(self):
        return self.camera.v
    
    @v.setter
    def v(self, newval):
        self.camera.v = newval


    def compute_dr_wrt(self, wrt):
        result = super(TexturedRenderer, self).compute_dr_wrt(wrt)
        
        if wrt is self.vc:
            cim = self.draw_color_image(with_vertex_colors=False).ravel()
            cim = sp.spdiags(row(cim), [0], cim.size, cim.size)
            result = cim.dot(result)
            
        return result

    def on_changed(self, which):
        super(TexturedRenderer, self).on_changed(which)

        # have to redo if frustum changes, b/c frustum triggers new context
        if 'texture_image' in which or 'frustum' in which:
            gl = self.glf
            texture_data = np.array(self.texture_image, dtype='uint8', order='C')
            tmp = np.zeros(2, dtype=np.uint32)
            
            self.release_textures()
            gl.GenTextures(1, tmp) # TODO: free after done
            self.textureID = tmp[0]

            gl.BindTexture(GL_TEXTURE_2D, self.textureID)

            gl.TexImage2Dub(GL_TEXTURE_2D, 0, GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL_BGR, texture_data.ravel())
            #gl.Hint(GL_GENERATE_MIPMAP_HINT, GL_NICEST) # must be GL_FASTEST, GL_NICEST or GL_DONT_CARE
            gl.GenerateMipmap(GL_TEXTURE_2D)


    @depends_on('vt', 'ft')
    def mesh_tex_coords(self):
        ftidxs = self.ft.ravel()
        data = np.asarray(self.vt[ftidxs].astype(np.float32)[:,0:2], np.float32, order='C')
        data[:,1] = 1.0 - 1.0*data[:,1]
        return data
    
    # Depends on 'f' because vpe/fpe depend on f
    @depends_on('vt', 'ft', 'f')
    def wireframe_tex_coords(self):
        vvt = np.zeros((self.v.r.size/3,2), dtype=np.float32, order='C')
        vvt[self.f.flatten()] = self.mesh_tex_coords
        edata = np.zeros((self.vpe.size,2), dtype=np.float32, order='C')
        edata = vvt[self.vpe.ravel()]
        return edata
    
    def texture_mapping_on(self, gl, with_vertex_colors):
        gl.Enable(GL_TEXTURE_2D)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        gl.TexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE if with_vertex_colors else GL_REPLACE);
        gl.BindTexture(GL_TEXTURE_2D, self.textureID)
        gl.EnableClientState(GL_TEXTURE_COORD_ARRAY)

    def texture_mapping_off(self, gl):
        gl.Disable(GL_TEXTURE_2D)
        gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)
    

    @depends_on(dterms+terms)
    def boundaryid_image(self):
        self._call_on_changed()
        self.texture_mapping_off(self.glb)
        result = draw_boundaryid_image(self.glb, self.v.r, self.f, self.vpe, self.fpe, self.camera)
        self.texture_mapping_on(self.glb, with_vertex_colors=True)
        return result


    @depends_on(terms+dterms)    
    def boundarycolor_image(self): 
        self._call_on_changed()
        try:
            gl = self.glf
            colors = self.vc.r.reshape((-1,3))[self.vpe.ravel()]
            gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            self.texture_mapping_on(gl, with_vertex_colors=False if colors is None else True)
            gl.TexCoordPointerf(2,0, self.wireframe_tex_coords.ravel())
            draw_colored_primitives(self.glf, self.v.r.reshape((-1,3)), self.vpe, colors)
            self.texture_mapping_off(gl)
            return np.asarray(deepcopy(gl.getImage()), np.float64)
        except:
            import pdb; pdb.set_trace()

    def draw_color_image(self, with_vertex_colors=True):
        self._call_on_changed()
        gl = self.glf
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if hasattr(self, 'background_image'):
            gl.MatrixMode(GL_PROJECTION)
            gl.PushMatrix()                
            gl.LoadIdentity()
            gl.Ortho(0, gl.width, 0, gl.height)
            gl.MatrixMode(GL_MODELVIEW)
            gl.PushMatrix()
            gl.LoadIdentity()
            gl.RasterPos2i(0, 0)
            
            gl.DrawPixels(self.background_image.shape[1],self.background_image.shape[0],GL_RGB,GL_FLOAT,np.asarray(self.background_image, np.float32))
            gl.PopMatrix() # modelview
            gl.MatrixMode(GL_PROJECTION)
            gl.PopMatrix() # projection
            assert (not gl.GetError())
            gl.Clear(GL_DEPTH_BUFFER_BIT)        


        self.texture_mapping_on(gl, with_vertex_colors)
        gl.TexCoordPointerf(2,0, self.mesh_tex_coords.ravel())
        
        colors = None
        if with_vertex_colors:
            colors = self.vc.r.reshape((-1,3))[self.f.ravel()]
        draw_colored_primitives(self.glf, self.v.r.reshape((-1,3)), self.f, colors)

        self.texture_mapping_off(gl)
        return np.asarray(deepcopy(gl.getImage()), np.float64)
        



def draw_edge_visibility(gl, v, e, f, hidden_wireframe=True):
    """Assumes camera is set up correctly in gl context."""
    gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ec = np.arange(1, len(e)+1)
    ec = np.tile(col(ec), (1, 3))
    ec[:, 0] = ec[:, 0] & 255
    ec[:, 1] = (ec[:, 1] >> 8 ) & 255
    ec[:, 2] = (ec[:, 2] >> 16 ) & 255
    ec = np.asarray(ec, dtype=np.uint8)
    
    draw_colored_primitives(gl, v, e, ec)
    
    if hidden_wireframe:
        gl.Enable(GL_POLYGON_OFFSET_FILL)
        gl.PolygonOffset(10.0, 1.0)
        draw_colored_primitives(gl, v, f, fc=np.zeros(f.shape))
        gl.Disable(GL_POLYGON_OFFSET_FILL)
    
    raw = np.asarray(gl.getImage(), np.uint32)
    raw = raw[:,:,0] + raw[:,:,1]*256 + raw[:,:,2]*256*256 - 1
    return raw

def draw_boundary_images(glf, glb, v, f, vpe, fpe, camera):
    """Assumes camera is set up correctly, and that glf has any texmapping on necessary."""
    glf.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glb.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    # Figure out which edges are on pairs of differently visible triangles
    from opendr.geometry import TriNormals
    tn = TriNormals(v, f).r.reshape((-1,3))
    campos = -cv2.Rodrigues(camera.rt.r)[0].T.dot(camera.t.r)
    rays_to_verts = v.reshape((-1,3)) - row(campos)
    rays_to_faces = rays_to_verts[f[:,0]] + rays_to_verts[f[:,1]] + rays_to_verts[f[:,2]]
    dps = np.sum(rays_to_faces * tn, axis=1)
    dps = dps[fpe[:,0]] * dps[fpe[:,1]]
    silhouette_edges = np.asarray(np.nonzero(dps<=0)[0], np.uint32)
    non_silhouette_edges = np.nonzero(dps>0)[0]
    lines_e = vpe[silhouette_edges]
    lines_v = v

    visibility = draw_edge_visibility(glb, lines_v, lines_e, f, hidden_wireframe=True)
    shape = visibility.shape
    visibility = visibility.ravel()
    visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    visibility[visible] = silhouette_edges[visibility[visible]]
    result = visibility.reshape(shape)
    return result


def compute_vpe_boundary_idxs(v, f, camera, fpe):
    # Figure out which edges are on pairs of differently visible triangles

    from opendr.geometry import TriNormals
    tn = TriNormals(v, f).r.reshape((-1,3))

    #ray = cv2.Rodrigues(camera.rt.r)[0].T[:,2]
    campos = -cv2.Rodrigues(camera.rt.r)[0].T.dot(camera.t.r)
    rays_to_verts = v.reshape((-1,3)) - row(campos)
    rays_to_faces = rays_to_verts[f[:,0]] + rays_to_verts[f[:,1]] + rays_to_verts[f[:,2]]
    faces_invisible = np.sum(rays_to_faces * tn, axis=1)
    dps = faces_invisible[fpe[:,0]] * faces_invisible[fpe[:,1]]
    silhouette_edges = np.asarray(np.nonzero(dps<=0)[0], np.uint32)
    return silhouette_edges, faces_invisible < 0



def draw_boundaryid_image(gl, v, f, vpe, fpe, camera):


    if False:
        visibility = draw_edge_visibility(gl, v, vpe, f, hidden_wireframe=True)
        return visibility
        
    if True:
    #try:
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        silhouette_edges, faces_facing_camera = compute_vpe_boundary_idxs(v, f, camera, fpe)
        lines_e = vpe[silhouette_edges]
        lines_v = v

        visibility = draw_edge_visibility(gl, lines_v, lines_e, f, hidden_wireframe=True)
        shape = visibility.shape
        visibility = visibility.ravel()
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        visibility[visible] = silhouette_edges[visibility[visible]]
        result = visibility.reshape(shape)
        return result

        #return np.asarray(deepcopy(gl.getImage()), np.float64)
    #except:
    #    import pdb; pdb.set_trace()



def setup_camera(gl, camera, frustum):
    _setup_camera(gl,
                  camera.c.r[0], camera.c.r[1],
                  camera.f.r[0], camera.f.r[1],
                  frustum['width'], frustum['height'],
                  frustum['near'], frustum['far'],
                  camera.view_matrix,
                  camera.k.r)
    
    

vs_source = """
#version 120

uniform float k1, k2, k3, k4, k5, k6;
uniform float p1, p2;
uniform float cx, cy, fx, fy;

void main()
{
    vec4 p0 = gl_ModelViewMatrix * gl_Vertex;

    float xp = p0[0] / p0[2];
    float yp = -p0[1] / p0[2];

    float r2 = xp*xp + yp*yp;
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    float m = (1.0 + k1*r2 + k2*r4 + k3*r6) / (1.0 + k4*r2 + k5*r4 + k6*r6);
    //p0[1] = -p0[1];
    p0[0] = xp * m + 2.*p1*xp*yp + p2*(r2+2*xp*xp);
    p0[1] = yp * m + p1*(r2+2*yp*yp) + 2.*p2*xp*yp;
    //p0[1] = -p0[1];
    p0[1] = -p0[1];

    gl_Position = gl_ProjectionMatrix * p0;
    //gl_Position = vec4(p0[0]*fx+cx, p0[1]*fy+cy, p0[2], p0[3]);
    //gl_Position[0] = p0[0]*fx+cx;
    //gl_Position[0] = p0[0];
    //gl_Position[0] = gl_Position[0] + 100;

    //----------------------------


    gl_FrontColor = gl_Color;
    gl_BackColor = gl_Color;

    //texture_coordinate = vec2(gl_MultiTexCoord0);
    gl_TexCoord[0] = gl_MultiTexCoord0;
}
"""

vs_source = """
#version 120

uniform float k1, k2, k3, k4, k5, k6;
uniform float p1, p2;

void main()
{
    vec4 p0 = gl_ModelViewMatrix * gl_Vertex;
    p0 = p0 / p0[3];

    float xp = -p0[0] / p0[2];
    float yp = p0[1] / p0[2];

    float r2 = xp*xp + yp*yp;
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    float m = (1.0 + k1*r2 + k2*r4 + k3*r6) / (1.0 + k4*r2 + k5*r4 + k6*r6);

    float xpp = m*xp + 2.*p1*xp*yp + p2*(r2+2*xp*xp);
    float ypp = m*yp + p1*(r2+2*yp*yp) + 2.*p2*xp*yp;

    p0[0] = -xpp * p0[2];
    p0[1] = ypp * p0[2];
    gl_Position = gl_ProjectionMatrix * p0;

    //----------------------------

    gl_FrontColor = gl_Color;
    gl_BackColor = gl_Color;

    //texture_coordinate = vec2(gl_MultiTexCoord0);
    gl_TexCoord[0] = gl_MultiTexCoord0;
}
"""


def _setup_camera(gl, cx, cy, fx, fy, w, h, near, far, view_matrix, k):
    k = np.asarray(k)
    gl.MatrixMode(GL_PROJECTION)
    gl.LoadIdentity();
    
    f = 0.5 * (fx + fy)
    right  =  (w-(cx+pixel_center_offset)) * (near/f)
    left   =           -(cx+pixel_center_offset)  * (near/f)
    top    = -(h-(cy+pixel_center_offset)) * (near/f)
    bottom =            (cy+pixel_center_offset)  * (near/f)
    gl.Frustum(left, right, bottom, top, near, far)

    gl.MatrixMode(GL_MODELVIEW);
    gl.LoadIdentity(); # I
    gl.Rotatef(180, 1, 0, 0) # I * xR(pi)

    view_mtx = np.asarray(np.vstack((view_matrix, np.array([0, 0, 0, 1]))), np.float32, order='F')
    gl.MultMatrixf(view_mtx) # I * xR(pi) * V

    gl.Enable(GL_DEPTH_TEST)
    gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    gl.Disable(GL_LIGHTING)
    gl.Disable(GL_CULL_FACE)
    gl.PixelStorei(GL_PACK_ALIGNMENT,1)
    gl.PixelStorei(GL_UNPACK_ALIGNMENT,1)

    if np.any(k):
        if not hasattr(gl, 'distortion_shader'):
            program = gl.CreateProgram()

            vs = gl.CreateShader(GL_VERTEX_SHADER)
            gl.ShaderSource(vs, 1, vs_source, len(vs_source))
            gl.AttachShader(program, vs)

            # fs = gl.CreateShader(GL_FRAGMENT_SHADER)
            # gl.ShaderSource(fs, 1, fs_source, len(fs_source))
            # gl.AttachShader(program, fs)

            gl.LinkProgram(program)
            gl.UseProgram(program)
            gl.distortion_shader = program

        gl.UseProgram(gl.distortion_shader)
        if len(k) != 8:
            tmp = k
            k = np.zeros(8)
            k[:len(tmp)] = tmp

        for idx, vname in enumerate(['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']):
            loc = gl.GetUniformLocation(gl.distortion_shader, vname)
            gl.Uniform1f(loc, k[idx])
    else:
        gl.UseProgram(0)


def draw_colored_verts(gl, v, f, vc):
    gl.EnableClientState(GL_VERTEX_ARRAY);
    gl.EnableClientState(GL_COLOR_ARRAY);
    gl.VertexPointer(np.ascontiguousarray(v).reshape((-1,3)));
    gl.ColorPointerd(np.ascontiguousarray(vc).reshape((-1,3)));
    gl.DrawElements(GL_TRIANGLES, np.asarray(f, np.uint32).ravel())
    
def draw_noncolored_verts(gl, v, f):
    gl.EnableClientState(GL_VERTEX_ARRAY);
    gl.DisableClientState(GL_COLOR_ARRAY);
    gl.VertexPointer(np.ascontiguousarray(v).reshape((-1,3)));
    gl.DrawElements(GL_TRIANGLES, np.asarray(f, np.uint32).ravel())

def main():
    pass

if __name__ == '__main__':
    main()

