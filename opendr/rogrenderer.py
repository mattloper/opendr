"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import sys
import os
from .cvwrap import cv2
import numpy as np

from opendr.renderer import TexturedRenderer
from opendr.lighting import LambertianPointLight
#from test_renderer import get_camera, convertCam, get_earthmesh


class RogTexRenderer(TexturedRenderer):
    terms = list(TexturedRenderer.terms) + ['sigma_big', 'sigma_small', 'bckground_mask']

    def draw_color_image(self, with_vertex_colors):
        result = super(self.__class__, self).draw_color_image(with_vertex_colors)
        return self.compute_rog(result)
        
    def compute_rog(self, im, is_real=False):
        if (self.sigma_small>=self.sigma_big) or (self.sigma_small==0): return im        
        if self.frustum['height'] != np.shape(im)[0]:
            im = cv2.resize(im, (self.frustum['width'], self.frustum['height'] ))
        if (len(im.shape)==3): im = np.mean(im, axis=2)
        im = np.asarray(im, np.float64)/255. 

        visibility = super(self.__class__, self).visibility_image
        im[visibility == 4294967295] = np.nan
             
        ksize = self.sigma_small*4
        if ksize % 2 == 0: ksize += 1
        smblur = cv2.GaussianBlur(im, (ksize, ksize), self.sigma_small)
        
        ksize = self.sigma_big*4
        if ksize % 2 == 0: ksize += 1
        bgblur = cv2.GaussianBlur(im, (ksize, ksize), self.sigma_big)
        
        # FIXME? zero values
        im = smblur / bgblur        
        
        if is_real:
            self.bckground_mask = np.ones_like(im); self.bckground_mask[np.isnan(im)] = np.nan

        im.ravel()[np.isnan(im.ravel())] = 0
        if hasattr(self, 'bckground_mask'): im.ravel()[np.isnan(self.bckground_mask.ravel())] = 0

        return im


        
"""def load_basics():
    np.random.seed(0)
    camera = get_camera(scaling=.4)    
    camera, frustum = convertCam(camera)
    mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([0,0,0]))
    
    lighting = LambertianPointLight(
        f=mesh.f, 
        num_verts=len(mesh.v), 
        light_pos=np.array([-1000,-1000,-1000]), 
        vc=mesh.vc, 
        light_color=np.array([1., 1., 1.]))
        
    bgcolor = np.array([0.,0.,0.])
    renderers = [
        TexturedRenderer(f=mesh.f, vc=mesh.vc, v=mesh.v, camera=camera, frustum=frustum, texture_image=mesh.texture_image, vt=mesh.vt, ft=mesh.ft, bgcolor=bgcolor),
        RogTexRenderer(f=mesh.f, vc=mesh.vc, v=mesh.v, camera=camera, frustum=frustum, texture_image=mesh.texture_image, vt=mesh.vt, ft=mesh.ft, bgcolor=bgcolor, sigma_small=3, sigma_big=5)]
    
    return mesh, lighting, camera, frustum, renderers
"""
        
if __name__ == '__main__':
   
    import matplotlib.pyplot as plt

    mesh, lighting, camera, frustum, renderers = load_basics()
    plt.ion()

    # Show default textured renderer
    plt.figure()
    plt.imshow(renderers[0].r)

    # Show ROG textured renderer
    plt.figure()    
    r2 = renderers[1].r.copy()
    r2 -= np.min(r2.ravel())
    r2 /= np.max(r2.ravel())
    plt.imshow(r2)
    import pdb; pdb.set_trace()
