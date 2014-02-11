#!/usr/bin/env python
# encoding: utf-8
"""
test_renderer.py

Created by Matthew Loper on 2013-03-22.
Copyright (c) 2013 MPI. All rights reserved.
"""

import unittest
import cv2
import time
import math
import numpy as np
import unittest
try:
    import matplotlib.pyplot as plt
    import matplotlib
except:
    from dummy import dummy as plt

from drender.renderer2 import *
import chumpy as ch
from chumpy import Ch
from chumpy.utils import row, col
from drender.lighting import *

from drender.test_dr.common import get_earthmesh, process

    
visualize = False


def getcam():
    from drender.camera import ProjectPoints3D

    w = 256
    h = 192

    f = np.array([200,200])
    rt = np.zeros(3)
    t = np.zeros(3)
    k = np.zeros(5)
    c = np.array([w/2., h/2.])

    pp = ProjectPoints3D(f=f, rt=rt, t=t, k=k, c=c)
    frustum = {'near': 1.0, 'far': 4.0, 'width': w, 'height': h}

    return pp, frustum



class TestRenderer(unittest.TestCase):

    def load_basics(self):
        np.random.seed(0)
        camera, frustum = getcam()

        frustum['far'] =  -100.
        frustum['near'] = 100.
        mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([0,0,0]))
        
        lighting = LambertianPointLight(
            f=mesh.f, 
            num_verts=len(mesh.v), 
            light_pos=np.array([-1000,-1000,-1000]), 
            vc=mesh.vc, 
            light_color=np.array([1., 1., 1.]))
        
            
        bgcolor = np.array([0.,0.,0.])
        renderers = [
            ColoredRenderer(f=mesh.f, camera=camera, frustum=frustum, bgcolor=bgcolor),
            TexturedRenderer(f=mesh.f, camera=camera, frustum=frustum, texture_image=np.asarray(mesh.texture_image*.9, np.float32)/255., vt=mesh.vt, ft=mesh.ft, bgcolor=bgcolor)]
        
        return mesh, lighting, camera, frustum, renderers
        
    
    @unittest.expectedFailure
    def test_cam_derivatives(self):
        mesh, lighting, camera, frustum, renderers = self.load_basics()


        if False:
            get = lambda x : x.k
            def set(k): camera.k = k
            eps = .001
        else:
            get = lambda x : x.f
            def set(k): camera.f = k
            eps = 1.

        for renderer in renderers:
            im_shape = (renderer.frustum['height'], renderer.frustum['width'], 3)

            # Render a rotating mesh
            mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        
            mesh_verts = Ch(mesh.v)  
            camera.set(v=mesh_verts)
            lighting.set(v=mesh_verts)
            renderer.set(vc=lighting)
            renderer.set(vc=mesh_verts.r.ravel()*0+1.)

            # Get pixels and derivatives
            r = renderer.r
            dr = renderer.dr_wrt(get(camera))
            
            # Establish a random direction
            direction = (np.random.rand(get(camera).r.size)-.5)*10.
            direction = direction * 0 + 3.

            # Render going forward in that direction
            set(get(camera).r+direction*eps/2.)
            renderer.set(camera=camera)
            rfwd = renderer.r
            # Render going backward in that direction
            set(get(camera).r-direction*eps/1.)
            renderer.set(camera=camera)
            rbwd = renderer.r
            set(get(camera).r+direction*eps/2.) # back to normal
            
            # Establish empirical and predicted derivatives
            dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)) / eps
            dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape) 

            images = {
                'shifted cam' : np.asarray(rfwd, np.float64)-.5,
                r'empirical cam $\left(\frac{dI}{dV}\right)$': dr_empirical,
                r'predicted cam $\left(\frac{dI}{dV}\right)$': dr_predicted,
                'difference cam': dr_predicted - dr_empirical,
                'image': r
            }

            dvflat = images['difference cam'].ravel()
            nonzero = dvflat[np.nonzero(dvflat!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                for idx, title in enumerate(sorted(images.keys(), reverse=True)):
                    plt.subplot(1,len(images.keys()), idx)
                    im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                    
                print 'cam: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'cam: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)
                plt.draw()
                plt.show()

            self.assertTrue(np.mean(np.abs(nonzero))<4e-2, msg="dr_predicted does not match dr_empirical (mean) for renderer %s" % renderer.__class__.__name__)
            self.assertTrue(np.median(np.abs(nonzero))<2e-2, msg="dr_predicted does not match dr_empirical (median) for renderer %s" % renderer.__class__.__name__)
        
        
    def test_vert_derivatives(self):
        #from body.misc.profdot import profdot
        #profdot('self._test_vert_derivatives()', globals(), locals())
        self._test_vert_derivatives()
    
    def _test_vert_derivatives(self):

        mesh, lighting, camera, frustum, renderers = self.load_basics()

        for renderer in renderers:

            im_shape = (renderer.frustum['height'], renderer.frustum['width'], 3)

            # Render a rotating mesh
            mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        
            mesh_verts = Ch(mesh.v)  
            camera.set(v=mesh_verts)
            lighting.set(v=mesh_verts)
            renderer.set(camera=camera)
            renderer.set(vc=lighting)

            # Get pixels and derivatives
            r = renderer.r
            dr = renderer.dr_wrt(mesh_verts)
            
            # Establish a random direction
            direction = (np.random.rand(mesh.v.size).reshape(mesh.v.shape)-.5)*.1 + np.sin(mesh.v*10)*.2
            direction *= .5
            eps = .2

            # Render going forward in that direction
            mesh_verts = Ch(mesh.v+direction*eps/2.)
            lighting.set(v=mesh_verts)
            renderer.set(v=mesh_verts, vc=lighting)
            rfwd = renderer.r
            
            # Render going backward in that direction
            mesh_verts = Ch(mesh.v-direction*eps/2.)
            lighting.set(v=mesh_verts)
            renderer.set(v=mesh_verts, vc=lighting)
            rbwd = renderer.r

            # Establish empirical and predicted derivatives
            dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)) / eps
            dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape)


            images = {
                'shifted verts' : np.asarray(rfwd, np.float64)-.5,
                r'empirical verts $\left(\frac{dI}{dV}\right)$': dr_empirical,
                r'predicted verts $\left(\frac{dI}{dV}\right)$': dr_predicted,
                'difference verts': dr_predicted - dr_empirical,
                'image': r
            }

            dvflat = images['difference verts'].ravel()
            nonzero = dvflat[np.nonzero(dvflat!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                for idx, title in enumerate(sorted(images.keys(), reverse=True)):
                    plt.subplot(1,len(images.keys()), idx)
                    im = process(images[title], vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                    
                print 'verts: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'verts: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)
                plt.draw()
                plt.show()

            self.assertTrue(np.mean(np.abs(nonzero))<7e-2, msg="dr_predicted does not match dr_empirical (mean) for renderer %s" % renderer.__class__.__name__)
            self.assertTrue(np.median(np.abs(nonzero))<4e-2, msg="dr_predicted does not match dr_empirical (median) for renderer %s" % renderer.__class__.__name__)
            

    def test_texture_derivatives(self, ):
        mesh, lighting, camera, frustum, renderers = self.load_basics()
        

        for renderer in renderers:
            if not hasattr(renderer, 'texture_image'):
                continue

            im_shape = (renderer.frustum['height'], renderer.frustum['width'], 3)

            # Render a rotating mesh
            #mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        
            mesh_verts = Ch(mesh.v)
            #camera.set(v=mesh_verts)
            #
            #
            light1_pos = Ch(np.array([-1000,-1000,-1000]))
            lighting.set(light_pos=light1_pos, v=mesh_verts)
            renderer.set(vc=lighting, v=mesh_verts)
            
            r = renderer.r
            dr = renderer.dr_wrt(renderer.texture_image).copy()            

            direction = (np.random.random(renderer.texture_image.r.size))
            texw = renderer.texture_image.shape[0]
            texh = renderer.texture_image.shape[0]
            xs = np.tile(row(np.arange(texw*3)/3), (texh, 1))
            ys = np.tile(col(np.arange(texh)), (1,texw*3))
            
            texim = renderer.texture_image.r.copy()

            direction = (np.round(xs/40.)%2) * (np.round(ys/40.)%2)*.2
            eps = 0.05
            
            renderer.texture_image = (texim.ravel() + direction.ravel()*eps).reshape(texim.shape)
            r_fwd = renderer.r
            
            
            if False:
                cv2.imshow('fjkdlsjf', renderer.texture_image.r)
                import pdb; pdb.set_trace()

            renderer.texture_image = (texim.ravel() - direction.ravel()*eps).reshape(texim.shape)
            r_bwd = renderer.r
            
            if False:
                cv2.imshow('fjkdlsjf', renderer.texture_image.r)
                import pdb; pdb.set_trace()
        
            # Establish empirical and predicted derivatives
            dr_empirical = (np.asarray(r_fwd, np.float64) - np.asarray(r_bwd, np.float64)) / eps / 2.
            
            dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape)

            images = {
                'shifted texim' : np.asarray(r_fwd, np.float64)-.5,
                r'empirical texim': dr_empirical,
                r'predicted texim': dr_predicted,
                'difference texim': dr_predicted-dr_empirical,
                'image': r
            }

            dvflat = images['difference texim'].ravel()
            nonzero = dvflat[np.nonzero(dvflat!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                for idx, title in enumerate(sorted(images.keys(), reverse=True)):
                    plt.subplot(1,len(images.keys()), idx)
                    im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                
                plt.show()
                print 'texim: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'texim: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)

            self.assertTrue(np.mean(np.abs(nonzero))<3e-2, msg="dr_predicted does not match dr_empirical (mean) for renderer %s" % renderer.__class__.__name__)
            self.assertTrue(np.median(np.abs(nonzero))<2.2e-2, msg="dr_predicted does not match dr_empirical (median) for renderer %s" % renderer.__class__.__name__)

    
    def test_lightpos_derivatives(self):
        
        mesh, lighting, camera, frustum, renderers = self.load_basics()
        

        for renderer in renderers:

            im_shape = (renderer.frustum['height'], renderer.frustum['width'], 3)

            # Render a rotating mesh
            mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))        
            mesh_verts = Ch(mesh.v)
            camera.set(v=mesh_verts)


            # Get predicted derivatives wrt light pos
            light1_pos = Ch(np.array([-1000,-1000,-1000]))
            lighting.set(light_pos=light1_pos, v=mesh_verts)
            renderer.set(vc=lighting, v=mesh_verts)
            
            r = renderer.r
            dr = renderer.dr_wrt(light1_pos).copy()            

            # Establish a random direction for the light
            direction = (np.random.rand(3)-.5)*1000.
            eps = 1.
        
            # Find empirical forward derivatives in that direction
            lighting.set(light_pos = light1_pos.r + direction*eps/2.)
            renderer.set(vc=lighting)
            rfwd = renderer.r
        
            # Find empirical backward derivatives in that direction
            lighting.set(light_pos = light1_pos.r - direction*eps/2.)
            renderer.set(vc=lighting)
            rbwd = renderer.r
        
            # Establish empirical and predicted derivatives
            dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)) / eps
            dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape)

            images = {
                'shifted lightpos' : np.asarray(rfwd, np.float64)-.5,
                r'empirical lightpos $\left(\frac{dI}{dL_p}\right)$': dr_empirical,
                r'predicted lightpos $\left(\frac{dI}{dL_p}\right)$': dr_predicted,
                'difference lightpos': dr_predicted-dr_empirical,
                'image': r
            }

            dvflat = images['difference lightpos'].ravel()
            nonzero = dvflat[np.nonzero(dvflat!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                for idx, title in enumerate(sorted(images.keys(), reverse=True)):
                    plt.subplot(1,len(images.keys()), idx)
                    im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                
                plt.show()
                print 'lightpos: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'lightpos: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)

            self.assertTrue(np.mean(np.abs(nonzero))<2.4e-2, msg="dr_predicted does not match dr_empirical (mean) for renderer %s" % renderer.__class__.__name__)
            self.assertTrue(np.median(np.abs(nonzero))<1.2e-2, msg="dr_predicted does not match dr_empirical (median) for renderer %s" % renderer.__class__.__name__)
            
        
        
    def test_color_derivatives(self):
        
        mesh, lighting, camera, frustum, renderers = self.load_basics()
        
        for renderer in renderers:

            im_shape = (renderer.frustum['height'], renderer.frustum['width'], 3)

            # Get pixels and dI/dC
            mesh = get_earthmesh(trans=np.array([0,0,5]), rotation = np.array([math.pi/2.,0,0]))

            mesh_verts = Ch(mesh.v)
            mesh_colors = Ch(mesh.vc)

            camera.set(v=mesh_verts)            

            # import pdb; pdb.set_trace()
            # print '-------------------------------------------'
            #lighting.set(vc=mesh_colors, v=mesh_verts)            
            lighting.vc = mesh_colors
            lighting.v = mesh_verts

            renderer.set(v=mesh_verts, vc=lighting)

            r = renderer.r
            dr = renderer.dr_wrt(mesh_colors).copy()

            # Establish a random direction
            eps = .4
            direction = (np.random.randn(mesh.v.size).reshape(mesh.v.shape)*.1 + np.sin(mesh.v*19)*.1).flatten()

            # Find empirical forward derivatives in that direction
            mesh_colors = Ch(mesh.vc.flatten()+direction*eps/2.)
            lighting.set(vc=mesh_colors)
            renderer.set(vc=lighting)
            rfwd = renderer.r

            # Find empirical backward derivatives in that direction
            mesh_colors = Ch(mesh.vc.flatten()-direction*eps/2.)
            lighting.set(vc=mesh_colors)
            renderer.set(vc=lighting)
            rbwd = renderer.r

            dr_empirical = (np.asarray(rfwd, np.float64) - np.asarray(rbwd, np.float64)) / eps
            dr_predicted = dr.dot(col(direction.flatten())).reshape(dr_empirical.shape)

            images = {
                'shifted colors' : np.asarray(rfwd, np.float64)-.5,
                r'empirical colors $\left(\frac{dI}{dC}\right)$': dr_empirical,
                r'predicted colors $\left(\frac{dI}{dC}\right)$': dr_predicted,
                'difference colors': dr_predicted-dr_empirical,
                'image': r
            }

            dvflat = images['difference colors'].ravel()
            nonzero = dvflat[np.nonzero(dvflat!=0)[0]]

            if visualize:
                matplotlib.rcParams.update({'font.size': 18})
                plt.figure(figsize=(6*3, 2*3))
                for idx, title in enumerate(sorted(images.keys(), reverse=True)):
                    plt.subplot(1,len(images.keys()), idx)
                    im = process(images[title].reshape(im_shape), vmin=-.5, vmax=.5)
                    plt.title(title)
                    plt.imshow(im)
                    
                plt.show()
                print 'color: median nonzero %.2e' % (np.median(np.abs(nonzero)),)
                print 'color: mean nonzero %.2e' % (np.mean(np.abs(nonzero)),)

            self.assertTrue(np.mean(np.abs(nonzero))<2e-2, msg="dr_predicted does not match dr_empirical (mean) for renderer %s" % renderer.__class__.__name__)
            self.assertTrue(np.median(np.abs(nonzero))<4.5e-3, msg="dr_predicted does not match dr_empirical (median) for renderer %s" % renderer.__class__.__name__)
                     

        


if __name__ == '__main__':
    plt.ion()
    visualize = True
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRenderer)
    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()
    import pdb; pdb.set_trace()

