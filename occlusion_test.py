from opendr.camera import ProjectPoints


import unittest


visualize = False

class TestOcclusion(unittest.TestCase):


    def test_occlusion(self):
        if visualize:
            import matplotlib.pyplot as plt
            plt.ion()

        # Create renderer
        import chumpy as ch
        import numpy as np
        from opendr.renderer import TexturedRenderer, ColoredRenderer
        #rn = TexturedRenderer()
        rn = ColoredRenderer()

        # Assign attributes to renderer
        from .util_tests import get_earthmesh
        m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
        rn.texture_image = m.texture_image
        rn.ft = m.ft
        rn.vt = m.vt
        m.v[:,2] = np.mean(m.v[:,2])

        # red is front and zero
        # green is back and 1
        t0 = ch.array([1,0,.1])
        t1 = ch.array([-1,0,.1])
        v0 = ch.array(m.v) + t0

        if False:
            v1 = ch.array(m.v*.4 + np.array([0,0,3.8])) + t1
        else:
            v1 = ch.array(m.v) + t1
        vc0 = v0*0 + np.array([[.4,0,0]])
        vc1 = v1*0 + np.array([[0,.4,0]])
        vc = ch.vstack((vc0, vc1))

        v = ch.vstack((v0, v1))
        f = np.vstack((m.f, m.f+len(v0)))

        w, h = (320, 240)
        rn.camera = ProjectPoints(v=v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
        rn.camera.t = ch.array([0,0,-2.5])
        rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
        m.vc = v.r*0 + np.array([[1,0,0]])
        rn.set(v=v, f=f, vc=vc)

        t0[:] = np.array([1.4, 0, .1-.02])
        t1[:] = np.array([-0.6, 0, .1+.02])

        target = rn.r

        if visualize:
            plt.figure()
            plt.imshow(target)
            plt.title('target')

            plt.figure()
            plt.show()

        im_orig = rn.r.copy()

        from .cvwrap import cv2

        tr = t0
        eps_emp = .02
        eps_pred = .02

        #blur = lambda x : cv2.blur(x, ksize=(5,5))
        blur = lambda x : x
        for tr in [t0, t1]:
            if tr is t0:
                sum_limits = np.array([2.1e+2, 6.9e+1, 1.6e+2])
            else:
                sum_limits = [1., 5., 4.]

            if visualize:
                plt.figure()
            for i in range(3):
                dr_pred = np.array(rn.dr_wrt(tr[i]).todense()).reshape(rn.shape) * eps_pred
                dr_pred = blur(dr_pred)

                # central differences
                tr[i] = tr[i].r + eps_emp/2.
                rn_greater = rn.r.copy()
                tr[i] = tr[i].r - eps_emp/1.
                rn_lesser = rn.r.copy()
                tr[i] = tr[i].r + eps_emp/2.

                dr_emp = blur((rn_greater - rn_lesser) * eps_pred / eps_emp)

                dr_pred_shown = np.clip(dr_pred, -.5, .5) + .5
                dr_emp_shown = np.clip(dr_emp, -.5, .5) + .5

                if visualize:
                    plt.subplot(3,3,i+1)
                    plt.imshow(dr_pred_shown)
                    plt.title('pred')
                    plt.axis('off')

                    plt.subplot(3,3,3+i+1)
                    plt.imshow(dr_emp_shown)
                    plt.title('empirical')
                    plt.axis('off')

                    plt.subplot(3,3,6+i+1)

                diff = np.abs(dr_emp - dr_pred)
                if visualize:
                    plt.imshow(diff)
                diff = diff.ravel()
                if visualize:
                    plt.title('diff (sum: %.2e)'  % (np.sum(diff)))
                    plt.axis('off')

                # print 'dr pred sum: %.2e' % (np.sum(np.abs(dr_pred.ravel())),)
                # print 'dr emp sum: %.2e' % (np.sum(np.abs(dr_emp.ravel())),)

                #import pdb; pdb.set_trace()
                self.assertTrue(np.sum(diff) < sum_limits[i])







if __name__ == '__main__':
    visualize = True
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOcclusion)
    unittest.TextTestRunner(verbosity=2).run(suite)
    import pdb; pdb.set_trace()
