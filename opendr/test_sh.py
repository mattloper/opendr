#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

from chumpy import Ch
import numpy as np
from chumpy.utils import row, col
from .lighting import SphericalHarmonics
import unittest
try:
    import matplotlib.pyplot as plt
except:
    from .dummy import dummy as plt
from .topology import loop_subdivider

visualize = False

def getcam():
    from .camera import ProjectPoints3D

    w = 640
    h = 320

    f = np.array([500,500])
    rt = np.zeros(3)
    t = np.zeros(3)
    k = np.zeros(5)
    c = np.array([w/2., h/2.])
    near = .1
    far = 20.

    frustum = {'near': near, 'far': far, 'width': int(w), 'height': int(h)}
    pp = ProjectPoints3D(f=f, rt=rt, t=t, k=k, c=c)

    return pp, frustum



class TestSphericalHarmonics(unittest.TestCase):

    def test_spherical_harmonics(self):
        global visualize
        if visualize:
            plt.ion()
    
        # Get mesh
        v, f = get_sphere_mesh()

        from .geometry import VertNormals
        vn = VertNormals(v=v, f=f)
        #vn =  Ch(mesh.estimate_vertex_normals())

        # Get camera
        cam, frustum = getcam()
    
        # Get renderer
        from .renderer import ColoredRenderer
        cam.v = v
        cr = ColoredRenderer(f=f, camera=cam, frustum=frustum, v=v)
    
        sh_red = SphericalHarmonics(vn=vn, light_color=np.array([1,0,0]))
        sh_green = SphericalHarmonics(vn=vn, light_color=np.array([0,1,0]))
    
        cr.vc = sh_red + sh_green
        
        ims_baseline = []
        for comp_idx, subplot_idx in enumerate([3,7,8,9,11,12,13,14,15]):
        
            sh_comps = np.zeros(9)
            sh_comps[comp_idx] = 1
            sh_red.components =  Ch(sh_comps)
            sh_green.components =  Ch(-sh_comps)
            
            newim = cr.r.reshape((frustum['height'], frustum['width'], 3))
            ims_baseline.append(newim)

            if visualize:
                plt.subplot(3,5,subplot_idx)
                plt.imshow(newim)
                plt.axis('off')
            
        offset = row(.4 * (np.random.rand(3)-.5))
        #offset = row(np.array([1.,1.,1.]))*.05
        vn_shifted = (vn.r + offset)
        vn_shifted = vn_shifted / col(np.sqrt(np.sum(vn_shifted**2, axis=1)))
        vn_shifted = vn_shifted.ravel()
        vn_shifted[vn_shifted>1.] = 1
        vn_shifted[vn_shifted<-1.] = -1
        vn_shifted = Ch(vn_shifted)
        cr.replace(sh_red.vn, vn_shifted)
        if True:
            for comp_idx in range(9):
                if visualize:
                    plt.figure(comp_idx+2)
        
                sh_comps = np.zeros(9)
                sh_comps[comp_idx] = 1
                sh_red.components =  Ch(sh_comps)
                sh_green.components =  Ch(-sh_comps)
        
                pred = cr.dr_wrt(vn_shifted).dot(col(vn_shifted.r.reshape(vn.r.shape) - vn.r)).reshape((frustum['height'], frustum['width'], 3))
                if visualize:
                    plt.subplot(1,2,1)
                    plt.imshow(pred)
                    plt.title('pred (comp %d)' % (comp_idx,))        
                    plt.subplot(1,2,2)
                    
                newim = cr.r.reshape((frustum['height'], frustum['width'], 3))
                emp = newim - ims_baseline[comp_idx]
                if visualize:
                    plt.imshow(emp)
                    plt.title('empirical (comp %d)' % (comp_idx,))
                pred_flat = pred.ravel()
                emp_flat = emp.ravel()
                nnz = np.unique(np.concatenate((np.nonzero(pred_flat)[0], np.nonzero(emp_flat)[0])))
                
                if comp_idx != 0:
                    med_diff = np.median(np.abs(pred_flat[nnz]-emp_flat[nnz]))
                    med_obs = np.median(np.abs(emp_flat[nnz]))
                    if comp_idx == 4 or comp_idx == 8:
                        self.assertTrue(med_diff / med_obs < .6)
                    else:
                        self.assertTrue(med_diff / med_obs < .3)
                if visualize:
                    plt.axis('off')
    

def get_sphere_mesh():
    from .util_tests import get_earthmesh

    mesh = get_earthmesh(np.zeros(3), np.zeros(3)) # load_mesh(filename)
    v, f = mesh.v*64., mesh.f

    for i in range(3):
        mtx, f = loop_subdivider(v, f)
        v = mtx.dot(v.ravel()).reshape((-1,3))
    v /= 200.
    v[:,2] += 2

    return v, f




if __name__ == '__main__':
    visualize = True
    plt.ion()
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSphericalHarmonics)
    unittest.TextTestRunner(verbosity=2).run(suite)
    plt.show()
    import pdb; pdb.set_trace()

