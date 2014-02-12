#!/usr/bin/env python

"""
Copyright (C) 2013
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import cv2
import numpy as np
from copy import deepcopy
import scipy.sparse as sp

import platform
if platform.system()=='Darwin':
    from .contexts.ctx_mac import OsContext
else:
    from .contexts.ctx_mesa import OsContext

from chumpy.utils import row, col
from .contexts._constants import *
from . import common
from .common import draw_visibility_image, draw_barycentric_image, draw_colored_primitives
from .topology import get_vertices_per_edge, get_faces_per_edge
from chumpy import Ch, depends_on

__all__ = ['ColoredRenderer', 'TexturedRenderer', 'DepthRenderer', 'BoundaryRenderer']

class BoundaryRenderer(Ch):
    terms = 'f', 'frustum', 'num_channels'
    dterms = 'camera'

    @property
    def v(self):
        return self.camera.v
    
    @v.setter
    def v(self, newval):
        self.camera.v = newval


    def compute_r(self):
        return self.color_image.ravel().reshape((self.frustum['height'], self.frustum['width'], self.num_channels))
    
    def compute_dr_wrt(self, wrt):
        if wrt is not self.camera:
            return None
        
        visibility = self.boundaryid_image
        shape = visibility.shape        
        
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        num_visible = len(visible)

        barycentric = self.barycentric_image
    
        result = common.dImage_wrt_2dVerts(self.color_image, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.vpe)
        
        result = result.tocoo()
        IS = result.row
        JS = result.col * 3 / 2
        shape = (result.shape[0], result.shape[1]*3/2)
        data = result.data
        #result.col = result.col * 3 / 2
        
        return sp.csc_matrix((data, (IS, JS)), shape=shape)

        

    def on_changed(self, which):

        if 'frustum' in which:
            w = self.frustum['width']
            h = self.frustum['height']
            self.glf = newCamera(w, h, GL_FLOAT, self.frustum['near'], self.frustum['far'])
            self.glb = newCamera(w, h, GL_UNSIGNED_BYTE, self.frustum['near'], self.frustum['far'])
            
        if 'v' in which and hasattr(self, 'camera') or 'camera' in which and hasattr(self, 'v'):
            self.camera.v = self.v
            
        if 'camera' in which:
            assert (hasattr(self.camera, 'z_coords')), "Need ProjectPoints3D, not ProjectPoints"


    @depends_on('f', 'frustum', 'camera', 'v')
    def boundarybool_image(self):
        boundaryid_image = self.boundaryid_image
        return np.asarray(boundaryid_image != 4294967295, np.float64).reshape(boundaryid_image.shape)


    @depends_on('f', 'frustum', 'camera', 'v')
    def boundaryid_image(self):
        return draw_boundaryid_image(self.glb, self.camera.r, self.f, self.vpe, self.fpe, self.camera)

    @depends_on('f', 'camera', 'v')
    def barycentric_image(self):
        return draw_barycentric_image(self.glf, self.camera.r, self.vpe)
    
    @depends_on('f')
    def primitives_per_edge(self):
        v = self.v.r.reshape((-1,3))
        f = self.f
        fpe = get_faces_per_edge(v, f)
        vpe = get_vertices_per_edge(v, f, fpe)
        return fpe, vpe

    @property
    def vpe(self):
        return self.primitives_per_edge[1]
    
    @property
    def fpe(self):
        return self.primitives_per_edge[0]
    
    @depends_on('f', 'camera', 'v', 'num_channels')
    def color_image(self):
        result = self.boundarybool_image        
        return np.dstack([result for i in range(self.num_channels)])




class ColoredRenderer(Ch):
    terms = 'f', 'frustum', 'background_image'
    dterms = 'vc', 'camera', 'bgcolor'        


    @property
    def v(self):
        return self.camera.v
    
    @v.setter
    def v(self, newval):
        self.camera.v = newval

    def compute_r(self):
        return self.color_image

    def on_changed(self, which):
        if 'frustum' in which:
            w = self.frustum['width']
            h = self.frustum['height']
            self.glf = newCamera(w, h, GL_FLOAT, self.frustum['near'], self.frustum['far'])
            self.glb = newCamera(w, h, GL_UNSIGNED_BYTE, self.frustum['near'], self.frustum['far'])
            
        if not hasattr(self, 'bgcolor'):
            self.bgcolor = Ch(np.array([.5,.5,.5]))
            which.add('bgcolor')
            
        if 'bgcolor' in which:
            self.glf.ClearColor(self.bgcolor.r[0], self.bgcolor.r[1], self.bgcolor.r[2], 1.)
            
        if 'v' in which:
            assert(self.v.r.shape[1] == 3)
            
        if 'camera' in which:
            assert (hasattr(self.camera, 'z_coords')), "Need ProjectPoints3D, not ProjectPoints"

            
            
    @depends_on('f', 'frustum', 'camera')
    def boundaryid_image(self):
        return draw_boundaryid_image(self.glb, self.camera.r, self.f, self.vpe, self.fpe, self.camera)
    
    @depends_on('f')
    def primitives_per_edge(self):
        v = self.v.r.reshape((-1,3))
        f = self.f
        fpe = get_faces_per_edge(v, f)
        vpe = get_vertices_per_edge(v, f, fpe)
        return fpe, vpe

    @property
    def vpe(self):
        return self.primitives_per_edge[1]
    
    @property
    def fpe(self):
        return self.primitives_per_edge[0]

    def flow_to(self, v_next, cam_next=None):
        return common.flow_to(self, v_next, cam_next)

    def compute_dr_wrt(self, wrt):
        if wrt not in (self.camera, self.vc, self.bgcolor):
            return None
            
        visibility = self.visibility_image
        shape = visibility.shape        
        color = self.color_image
        
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        num_visible = len(visible)

        barycentric = self.barycentric_image
    
    
        if wrt is self.camera:
                        
            result = common.dImage_wrt_2dVerts(
                color, visible, visibility, barycentric,
                self.frustum['width'], self.frustum['height'],
                self.v.size/3, self.f)
            result = result.tocoo()
            IS = result.row
            JS = result.col * 3 / 2
            shape = (result.shape[0], result.shape[1]*3/2)
            data = result.data
            #result.col = result.col * 3 / 2
            
            return sp.csc_matrix((data, (IS, JS)), shape=shape)
        
        elif wrt is self.vc:
            return common.dr_wrt_vc(visible, visibility, self.f, barycentric, self.frustum, self.v.size)

        elif wrt is self.bgcolor:
            return common.dr_wrt_bgcolor(visibility, self.frustum)

    @depends_on('f', 'frustum', 'camera')
    def visibility_image(self):
        return draw_visibility_image(self.glb, self.camera.r, self.f)

    @depends_on('f', 'camera')
    def barycentric_image(self):
        return draw_barycentric_image(self.glf, self.camera.r, self.f)

    @depends_on(terms + dterms)
    def color_image(self):
        gl = self.glf
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        if hasattr(self, 'background_image'):
            gl.RasterPos2i(0, 0)
            
            gl.DrawPixels(self.background_image.shape[1],self.background_image.shape[0],GL_RGB,GL_FLOAT,np.asarray(self.background_image, np.float32))
            assert (not gl.GetError())
            gl.Clear(GL_DEPTH_BUFFER_BIT)        

        # use face colors if given
        draw_colored_verts(gl, self.camera.r, self.f, self.vc.r)

        return np.asarray(deepcopy(gl.getImage()), np.float64)
       
       
class DepthRenderer(Ch):
    terms = 'f', 'frustum', 'background_image', 'inplane_only'
    dterms = 'camera', 'v'        
    
    def compute_r(self):
        return self.depth_image
        
    def compute_dr_wrt(self, wrt):
        
        if wrt is not self.camera:
            return None
        
        visibility = self.visibility_image
        visible = np.nonzero(visibility.ravel() != 4294967295)[0]
        if wrt is self.camera:
            
            shape = visibility.shape        
            depth = self.depth_image
            
            barycentric = self.barycentric_image
    
            result_inplane = common.dImage_wrt_2dVerts(depth, visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)
            result_inplane = result_inplane.tocoo()
            inplane_IS = result_inplane.row
            inplane_JS = result_inplane.col * 3 / 2
            inplane_data = result_inplane.data
            shape = (result_inplane.shape[0], result_inplane.shape[1]*3/2)

            if self.inplane_only:
                IS = inplane_IS
                JS = inplane_JS
                data = inplane_data
            else:
                result_depth = dImage_wrt_depth(visible, visibility, barycentric, self.frustum['width'], self.frustum['height'], self.v.r.size/3, self.f)
    
                result_depth = result_depth.tocoo()
                depth_IS = result_depth.row
                depth_JS = result_depth.col * 3 + 2
                depth_data = result_depth.data
                
                IS = np.concatenate((inplane_IS, depth_IS))
                JS = np.concatenate((inplane_JS, depth_JS))
                data = np.concatenate((inplane_data, depth_data))
            
            result = sp.csc_matrix((data, (IS, JS)), shape=shape)
            return result            
    
    def on_changed(self, which):
            
        if 'frustum' in which or 'camera' in which:
            w = self.frustum['width']
            h = self.frustum['height']
            self.glf = newCamera(w, h, GL_FLOAT, self.frustum['near'], self.frustum['far'])
            self.glb = newCamera(w, h, GL_UNSIGNED_BYTE, self.frustum['near'], self.frustum['far'])
            
        if not hasattr(self, 'inplane_only'):
            self.inplane_only=False
            
        if 'background_image' in which:
            assert (not np.any(np.isnan(self.background_image.ravel()))), 'background_image should not contain nans'
            assert (not np.any(np.isinf(self.background_image.ravel()))), 'background_image should not contain infs'
            
        assert(self.v is self.camera.v)
        
        if 'camera' in which:
            assert (hasattr(self.camera, 'z_coords')), "Need ProjectPoints3D, not ProjectPoints"

            
        #if ('v' in which and hasattr(self, 'camera')) or ('camera' in which and hasattr(self, 'v')):
        #    self.camera.v = self.v
            
        #if ('v' in which and hasattr(self, 'camera')): 
        #    self.camera.v = self.v
        #else:
        #    assert(self.v is self.camera.v)

    @depends_on('f', 'camera', 'background_image')
    def depth_image(self):
        gl = self.glf
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # use face colors if given
        draw_noncolored_verts(gl, self.camera.r, self.f)

        result = np.asarray(deepcopy(-gl.getDepth()), np.float64)
        if hasattr(self, 'background_image'):            
            visibility = self.visibility_image
            visible = np.nonzero(visibility.ravel() != 4294967295)[0]
            result2 = np.require(self.background_image.copy(), dtype=np.float64)
            result2.ravel()[visible] = result.ravel()[visible]
            result = result2
        
        return result
        

    @depends_on('f', 'camera')
    def barycentric_image(self):
        return draw_barycentric_image(self.glf, self.camera.r, self.f)
    
    @depends_on('f', 'camera')
    def visibility_image(self):
        return draw_visibility_image(self.glb, self.camera.r, self.f)
    
    def getDepthMesh(self, depth_image=None):
        self._call_on_changed() # make everything is up-to-date            
        
        #v = self.glf.getDepthCloud(depth_image)
        w = self.frustum['width']
        h = self.frustum['height']
        idxs = np.arange(w*h).reshape((h, w))
        
        if True:
            xs = idxs % w
            ys = idxs / w
            
            pts = np.asarray(np.hstack((col(xs), col(ys))), dtype=np.float32).reshape((-1,1,2))
            cam_mtx = np.asarray(self.camera.camera_mtx, dtype=np.float32)

            try:
                udp = cv2.undistortPoints(pts, self.camera.camera_mtx, self.camera.k.r).reshape((-1,2))
            except:
                assert(np.max(np.abs(self.camera.k.r)) == 0)
                pts_homog = np.vstack((row(xs), row(ys), row(xs*0+1)))
                pts_screen = np.linalg.inv(cam_mtx).dot(pts_homog)
                pts_screen = pts_screen[:2,:] / row(pts_screen[2,:])
                udp = pts_screen.T.copy()

            udp = udp * col(depth_image) 
            udp = np.hstack((udp, col(depth_image)))
            v = udp

    
        # v0 is upperleft, v1 is upper right, v2 is lowerleft, v3 is lowerright
        v0 = col(idxs[:-1,:-1])
        v1 = col(idxs[:-1,1:])
        v2 = col(idxs[1:,:-1])
        v3 = col(idxs[1:,1:])

        f = np.hstack((v0, v1, v2, v1, v3, v2)).reshape((-1,3))
        v[:,1] *= -1
        v[:,2] *= -1
        return v, f

       
class TexturedRenderer(ColoredRenderer):
    terms = 'f', 'frustum', 'vt', 'ft', 'background_image'
    dterms = 'vc', 'camera', 'bgcolor', 'texture_image'


    def __del__(self):
        self.release_textures()
        
    def release_textures(self):
        if hasattr(self, 'textureID'):
            arr = np.asarray(np.array([self.textureID]), np.uint32, order='C')
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
        elif wrt is self.texture_image:
            IS = np.nonzero(self.visibility_image.ravel() != 4294967295)[0]
            JS = self.texcoord_image_quantized.ravel()[IS]
            
            clr_im = self.draw_color_image(with_vertex_colors=True, with_texmapping=False)
            
            if False:            
                cv2.imshow('clr_im', clr_im)
                cv2.imshow('texmap', self.texture_image.r)
                cv2.waitKey(1)

            r = clr_im[:,:,0].ravel()[IS]
            g = clr_im[:,:,1].ravel()[IS]
            b = clr_im[:,:,2].ravel()[IS]
            data = np.concatenate((r,g,b))

            IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
            JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
            
            
            return sp.csc_matrix((data, (IS, JS)), shape=(self.r.size, wrt.r.size))
            
        return result

    def on_changed(self, which):
        super(TexturedRenderer, self).on_changed(which)

        if 'ft' in which or 'f' in which:
            assert(self.ft.size == self.f.size)
            
        # have to redo if frustum changes, b/c frustum triggers new context
        if 'texture_image' in which or 'frustum' in which:
            gl = self.glf
            texture_data = self.texture_image.r
            assert(isinstance(texture_data.ravel()[0], np.float32) or isinstance(texture_data.ravel()[0], np.float64))

            texture_data = np.asarray(self.texture_image.r, dtype='float32', order='C')
            self.release_textures()
            tmp = np.zeros(2, dtype=np.uint32)
            
            gl.GenTextures(1, tmp) # TODO: free after done
            self.textureID = tmp[0]

            gl.PixelStorei(GL_UNPACK_ALIGNMENT,1)
            gl.BindTexture(GL_TEXTURE_2D, self.textureID)

            gl.TexImage2Df(GL_TEXTURE_2D, 0, GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL_RGB, texture_data.ravel())
            #gl.TexImage2Dub(GL_TEXTURE_2D, 0, GL_RGB, texture_data.shape[1], texture_data.shape[0], 0, GL_RGB, texture_data.ravel())
            #gl.Hint(GL_GENERATE_MIPMAP_HINT, GL_NICEST) # must be GL_FASTEST, GL_NICEST or GL_DONT_CARE
            gl.GenerateMipmap(GL_TEXTURE_2D)

    @depends_on('vt', 'ft', 'f', 'frustum', 'camera')
    def texcoord_image_quantized(self):
        texcoord_image = self.texcoord_image.copy()
        texcoord_image[:,:,0] *= self.texture_image.shape[1]-1
        texcoord_image[:,:,1] *= self.texture_image.shape[0]-1
        texcoord_image = np.round(texcoord_image)
        texcoord_image = texcoord_image[:,:,0] + texcoord_image[:,:,1]*self.texture_image.shape[1]
        return texcoord_image
    

    @depends_on('vt', 'ft', 'f', 'frustum', 'camera')
    def texcoord_image(self):
        gl = self.glf
        self.texture_mapping_off(gl)
        gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # want vtc: texture-coordinates per vertex (not per element in vc)
        colors = self.vt[self.ft.ravel()]
        #vtc = np.zeros((len(self.camera.r), 2))
        #for idx, vidx in enumerate(self.f.ravel()):
        #    tidx = self.ft.ravel()[idx]
        #    vtc[vidx] = self.vt[tidx]
        
        colors = np.asarray(np.hstack((colors, col(colors[:,0]*0))), np.float64, order='C')
        draw_colored_primitives(gl, self.camera.r.reshape((-1,3)), self.f, colors)
        result = np.asarray(deepcopy(gl.getImage()), np.float64, order='C')[:,:,:2].copy()
        result[:,:,1] = 1. - result[:,:,1] 
        return result
    

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
    
    def texture_mapping_on(self, gl):
        gl.Enable(GL_TEXTURE_2D)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        gl.TexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        gl.TexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        gl.BindTexture(GL_TEXTURE_2D, self.textureID)
        gl.EnableClientState(GL_TEXTURE_COORD_ARRAY)

    def texture_mapping_off(self, gl):
        gl.Disable(GL_TEXTURE_2D)
        gl.DisableClientState(GL_TEXTURE_COORD_ARRAY)
    

    @depends_on(terms+dterms)    
    def boundarycolor_image(self): 
        try:
            gl = self.glf
            colors = self.vc.r.reshape((-1,3))[self.vpe.ravel()]
            gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            self.texture_mapping_on(gl)
            gl.TexCoordPointerf(2,0, self.wireframe_tex_coords.ravel())
            draw_colored_primitives(self.glf, self.camera.r.reshape((-1,3)), self.vpe, colors)
            self.texture_mapping_off(gl)
            return np.asarray(deepcopy(gl.getImage()), np.float64, order='C')
        except:
            import pdb; pdb.set_trace()

    def draw_color_image(self, with_vertex_colors, with_texmapping=True):
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
            
            px = np.asarray(self.background_image, np.float32, order='C')
            gl.DrawPixels(self.background_image.shape[1],self.background_image.shape[0],GL_RGB,GL_FLOAT,px)
            gl.PopMatrix() # modelview
            gl.MatrixMode(GL_PROJECTION)
            gl.PopMatrix() # projection
            assert (not gl.GetError())
            gl.Clear(GL_DEPTH_BUFFER_BIT)        

        if with_texmapping:
            self.texture_mapping_on(gl)
        else:
            self.texture_mapping_off(gl)
        gl.TexCoordPointerf(2,0, self.mesh_tex_coords.ravel())
        
        colors = None
        if with_vertex_colors:
            colors = self.vc.r.reshape((-1,3))[self.f.ravel()]
        draw_colored_primitives(self.glf, self.camera.r.reshape((-1,3)), self.f, colors)

        self.texture_mapping_off(gl)
        return np.asarray(deepcopy(gl.getImage()), np.float64, order='C')
        

    @depends_on(terms+dterms)    
    def color_image(self):
        return self.draw_color_image(with_vertex_colors=True)
 
def dImage_wrt_depth(visible, visibility, barycentric, image_width, image_height, num_verts, f):

    shape = visibility.shape        

    # Step 1: get the structure ready, ie the IS and the JS
    IS = np.tile(col(visible), (1, f.shape[1])).ravel()
    JS = col(f[visibility.ravel()[visible]].ravel())


    pxs = np.asarray(visible % shape[1], np.int32)
    pys = np.asarray(np.floor(np.floor(visible) / shape[1]), np.int32)

    datas = []

    # The data is weighted according to barycentric coordinates
    bc0 = col(barycentric[pys, pxs, 0])   
    bc1 = col(barycentric[pys, pxs, 1])  
    bc2 = col(barycentric[pys, pxs, 2])
    
    datas.append(np.hstack((bc0,bc1,bc2)).ravel())
            
    data = np.concatenate(datas)

    result = sp.csc_matrix((data, (IS.ravel(), JS.ravel())), shape=(image_width*image_height, num_verts))
    return result

        


def compute_vpe_boundary_idxs(v, f, camera, fpe):
    # Figure out which edges are on pairs of differently visible triangles
    
    from opendr.geometry import TriNormals

    if False:
        tn = TriNormals(v, f).r.reshape((-1,3))
        #ray = cv2.Rodrigues(camera.rt.r)[0].T[:,2]
        campos = -cv2.Rodrigues(camera.rt.r)[0].T.dot(camera.t.r)
        rays_to_verts = v.reshape((-1,3)) - row(campos)
        rays_to_faces = rays_to_verts[f[:,0]] + rays_to_verts[f[:,1]] + rays_to_verts[f[:,2]]
        dps = np.sum(rays_to_faces * tn, axis=1)
        dps = dps[fpe[:,0]] * dps[fpe[:,1]]
        silhouette_edges = np.asarray(np.nonzero(dps<=0)[0], np.uint32)
        return silhouette_edges
    else:
        tn = TriNormals(v, f).r.reshape((-1,3))
        dps = np.asarray(np.nonzero(np.sign(tn[fpe[:,0],2]) != np.sign(tn[fpe[:,1],2]))[0], np.uint32)
        return dps

def draw_boundaryid_image(gl, v, f, vpe, fpe, camera):
    gl.Clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    silhouette_edges = compute_vpe_boundary_idxs(v, f, camera, fpe)
    lines_e = vpe[silhouette_edges]
    lines_v = v

    visibility = draw_edge_visibility(gl, lines_v, lines_e, f, hidden_wireframe=True)
    shape = visibility.shape
    visibility = visibility.ravel()
    visible = np.nonzero(visibility.ravel() != 4294967295)[0]
    visibility[visible] = silhouette_edges[visibility[visible]]
    result = visibility.reshape(shape)
    return result


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
        gl.PolygonOffset(5.0, 1.)
        draw_colored_primitives(gl, v, f, fc=np.zeros(f.shape))
        gl.Disable(GL_POLYGON_OFFSET_FILL)
    
    raw = np.asarray(gl.getImage(), np.uint32)
    raw = raw[:,:,0] + raw[:,:,1]*256 + raw[:,:,2]*256*256 - 1
    return raw



def newCamera(w, h, typ, near, far):            
    gl = OsContext(w, h, typ=typ)
    gl.Viewport(0, 0, w, h)    
    gl.MatrixMode(GL_PROJECTION)
    gl.LoadIdentity()
    gl.Ortho(0, gl.width, 0, gl.height, near, far)
    gl.MatrixMode(GL_MODELVIEW)
    gl.LoadIdentity()
    
    gl.Enable(GL_DEPTH_TEST)
    gl.PolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    gl.Disable(GL_LIGHTING);
    gl.Disable(GL_CULL_FACE)    
    gl.PixelStorei(GL_PACK_ALIGNMENT,1)
    gl.PixelStorei(GL_UNPACK_ALIGNMENT,1)
    return gl

def draw_colored_verts(gl, v, f, vc):
    gl.EnableClientState(GL_VERTEX_ARRAY);
    gl.EnableClientState(GL_COLOR_ARRAY);
    v = np.asarray(v, dtype=np.float64, order='C')
    gl.VertexPointer(v)
    vc = np.asarray(vc.reshape((-1,3)), dtype=np.float64, order='C')
    gl.ColorPointerd(vc);
    f = np.asarray(f, np.uint32, order='C').ravel()
    gl.DrawElements(GL_TRIANGLES, f)

def draw_noncolored_verts(gl, v, f):
    gl.EnableClientState(GL_VERTEX_ARRAY);
    gl.DisableClientState(GL_COLOR_ARRAY);
    v = np.asarray(v.reshape((-1,3)), dtype=np.float64, order='C')
    gl.VertexPointer(v)
    f = np.asarray(f, np.uint32, order='C').ravel()
    gl.DrawElements(GL_TRIANGLES, f)
