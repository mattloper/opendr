#!/usr/bin/env python
# encoding: utf-8
"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

import sys
import os
import unittest
import chumpy as ch
from chumpy import Ch
import numpy as np
from .util_tests import get_earthmesh

class TestGeometry(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_rodrigues(self):
        from .geometry import Rodrigues
        rt = np.random.randn(3)
        rt2 = rt + np.random.rand(3)*1e-5
        foo1 = Rodrigues(rt = rt)
        foo2 = Rodrigues(rt = rt2)
        
        empirical = (foo2.r - foo1.r).flatten()
        predicted = foo1.dr_wrt(foo1.rt).dot(rt2-rt)
        
        self.assertTrue(np.max(np.abs(empirical - predicted)) < 1e-10)
        

    def test_vert_normals(self):
        from .geometry import VertNormals
        import numpy as np

        mesh = get_earthmesh(np.zeros(3), np.zeros(3))
        v, f = mesh.v*127., mesh.f

        vn1 = VertNormals(f=f, v=v)
        dr_predicted = vn1.dr_wrt(vn1.v).copy()

        eps = .00001 * np.random.randn(v.size).reshape(v.shape)
        v += eps
        vn2 = VertNormals(v=v, f=f)
        empirical_diff = (vn2.r - vn1.r).reshape((-1,3))
        
        predicted_diff = dr_predicted.dot(eps.flatten()).reshape((-1,3))   

        if False:
            print(np.max(np.abs(empirical_diff-predicted_diff)))
            print(empirical_diff[:6])
            print(predicted_diff[:6])
        self.assertTrue(np.max(np.abs(empirical_diff-predicted_diff)) < 6e-13)



suite = unittest.TestLoader().loadTestsFromTestCase(TestGeometry)

if __name__ == '__main__':
    unittest.main()
