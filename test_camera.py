#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import unittest
import numpy as np

from .camera import *
import chumpy as ch




class TestCamera(unittest.TestCase):
    
    def get_cam_params(self):

        v_raw = np.sin(np.arange(900)).reshape((-1,3))
        v_raw[:, 2] -= 2
        
        rt = ch.zeros(3)
        t = ch.zeros(3)
        f = ch.array([500,500])
        c = ch.array([320,240])
        k = ch.zeros(5)

        cam_params = {'v': ch.Ch(v_raw), 'rt': rt, 't': t, 'f': f, 'c': c, 'k': k}
        return cam_params
    
    def test_project_points(self):
        self.project_points(ProjectPoints)
        self.project_points(ProjectPoints3D)
        
    def project_points(self, cls):
        cam_params = self.get_cam_params()
        for key, value in list(cam_params.items()):
            
            eps = (np.random.random(value.r.size)-.5) * 1e-5
            pp_dist = cls(**cam_params)
                        
            old_val = pp_dist.r.copy()            
            old_dr = pp_dist.dr_wrt(value).dot(eps)
    
            tmp = cam_params[key].r.copy()
            tmp += eps.reshape(tmp.shape)

            cam_params[key] = ch.Ch(tmp)
            diff = ((cls(**cam_params).r - old_val))

            raw_dr_diff = np.abs(old_dr.flatten() - diff.flatten())
            med_diff = np.median(raw_dr_diff)
            max_diff = np.max(raw_dr_diff)
            
            #pct_diff = (100. * max_diff / np.mean(np.abs(old_val.flatten())))
            # print 'testing for %s' % (key,)
            # print 'empirical' + str(diff.flatten()[:5])
            # print 'predicted' + str(old_dr[:5])
            # print 'med diff: %.2e' % (med_diff,)
            # print 'max diff: %.2e' % (max_diff,)
            #print 'pct diff: %.2e%%' % (pct_diff,)

            self.assertLess(med_diff, 1e-8)
            self.assertLess(max_diff, 5e-8)
            

        pp_dist = cls(**cam_params)
        
        # Test to make sure that depends_on is working
        for name in ('rt', 't', 'f', 'c', 'k'):
            aa = pp_dist.camera_mtx
            setattr(pp_dist, name, getattr(pp_dist, name).r + 1)
            bb = pp_dist.camera_mtx

            if name in ('f', 'c'):
                self.assertTrue(aa is not bb)
            else:
                self.assertTrue(aa is bb)


if __name__ == '__main__':
    unittest.main()

