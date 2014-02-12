from opendr.renderer import DepthRenderer
import numpy as np
import unittest


visualize = False

class TestDepthRenderer(unittest.TestCase):

    def test_depth_renderer(self):
        # Create renderer
        import chumpy as ch
        from opendr.renderer import DepthRenderer
        rn = DepthRenderer()

        # Assign attributes to renderer
        from opendr.test_dr.common import get_earthmesh
        m = get_earthmesh(trans=ch.array([0,0,0]), rotation=ch.zeros(3))
        m.v = m.v * .01
        m.v[:,2] += 4
        w, h = (320, 240)
        from opendr.camera import ProjectPoints
        rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
        rn.set(v=m.v, f=m.f, vc=m.vc*0+1, bgcolor=ch.zeros(3))
        
        # import time
        # tm = time.time()
        # rn.r
        # print 'took %es' % (time.time() - tm)

        # Show it
        if visualize:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.imshow(rn.r)
            plt.show()

        # print np.min(rn.r.ravel())
        # print np.max(rn.r.ravel())
        self.assertTrue(np.abs(np.min(rn.r.ravel()) - 3.98) < 1e-5)
        self.assertTrue(np.abs(np.min(m.v[:,2]) - np.min(rn.r.ravel())) < 1e-5)
        self.assertTrue(np.abs(rn.r[h/2,w/2] - 3.98) < 1e-5)

if __name__ == '__main__':
    visualize = True
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDepthRenderer)
    unittest.TextTestRunner(verbosity=2).run(suite)
    import pdb; pdb.set_trace()
