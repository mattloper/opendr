from .renderer import DepthRenderer
import numpy as np
import unittest


visualize = False

class TestDepthRenderer(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_depth_image(self):
        # Create renderer
        import chumpy as ch
        from .renderer import DepthRenderer
        rn = DepthRenderer()

        # Assign attributes to renderer
        from .util_tests import get_earthmesh
        m = get_earthmesh(trans=ch.array([0,0,0]), rotation=ch.zeros(3))
        m.v = m.v * .01
        m.v[:,2] += 4
        w, h = (320, 240)
        from .camera import ProjectPoints
        rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
        rn.set(v=m.v, f=m.f, vc=m.vc*0+1, bgcolor=ch.zeros(3))
        
        # import time
        # tm = time.time()
        # rn.r
        # print 'took %es' % (time.time() - tm)

        # print np.min(rn.r.ravel())
        # print np.max(rn.r.ravel())
        self.assertLess(np.abs(np.min(rn.r.ravel()) - 3.98), 1e-5)
        self.assertLess(np.abs(np.min(m.v[:,2]) - np.min(rn.r.ravel())), 1e-5)
        self.assertLess(np.abs(rn.r[h/2,w/2] - 3.98), 1e-5)

    def test_derivatives(self):
        import chumpy as ch
        from chumpy.utils import row
        import numpy as np
        from .renderer import DepthRenderer

        rn = DepthRenderer()

        # Assign attributes to renderer
        from .util_tests import get_earthmesh
        m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
        w, h = (320, 240)
        from .camera import ProjectPoints
        rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
        rn.set(v=m.v, f=m.f, bgcolor=ch.zeros(3))

        if visualize:
            import matplotlib.pyplot as plt
            plt.figure()
        for which in range(3):
            r1 = rn.r

            adder = np.zeros(3)
            adder[which] = .01
            change = rn.v.r * 0 + row(adder)
            dr_pred = rn.dr_wrt(rn.v).dot(change.ravel()).reshape(rn.shape)
            rn.v = rn.v.r + change

            r2 = rn.r
            dr_emp = r2 - r1

            # print np.mean(np.abs(dr_pred-dr_emp))

            self.assertLess(np.mean(np.abs(dr_pred-dr_emp)), .031)

            if visualize:
                plt.subplot(2,3,which+1)
                plt.imshow(dr_pred)
                plt.clim(-.01,.01)
                plt.title('emp')
                plt.subplot(2,3,which+4)
                plt.imshow(dr_emp)
                plt.clim(-.01,.01)
                plt.title('pred')

    def test_derivatives2(self):
        import chumpy as ch
        import numpy as np
        from .renderer import DepthRenderer

        rn = DepthRenderer()

        # Assign attributes to renderer
        from .util_tests import get_earthmesh
        m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
        w, h = (320, 240)
        from .camera import ProjectPoints
        rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
        rn.set(v=m.v, f=m.f, bgcolor=ch.zeros(3))

        if visualize:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.figure()

        for which in range(3):
            r1 = rn.r

            adder = np.random.rand(rn.v.r.size).reshape(rn.v.r.shape)*.01
            change = rn.v.r * 0 + adder
            dr_pred = rn.dr_wrt(rn.v).dot(change.ravel()).reshape(rn.shape)
            rn.v = rn.v.r + change

            r2 = rn.r
            dr_emp = r2 - r1

            #print np.mean(np.abs(dr_pred-dr_emp))

            self.assertLess(np.mean(np.abs(dr_pred-dr_emp)), .024)

            if visualize:
                plt.subplot(2,3,which+1)
                plt.imshow(dr_pred)
                plt.clim(-.01,.01)
                plt.title('emp')
                plt.subplot(2,3,which+4)
                plt.imshow(dr_emp)
                plt.clim(-.01,.01)
                plt.title('pred')
                plt.draw()
                plt.show()

if __name__ == '__main__':
    visualize = True
    #unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDepthRenderer)
    unittest.TextTestRunner(verbosity=2).run(suite)
    import pdb; pdb.set_trace()
