"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""
__all__ = ['mstack', 'wget']

def mstack(vs, fs):
    import chumpy as ch
    import numpy as np
    lengths = [v.shape[0] for v in vs]
    f = np.vstack([fs[i]+np.sum(lengths[:i]).astype(np.uint32) for i in range(len(fs))])
    v = ch.vstack(vs)

    return v, f


def wget(url, dest_fname=None):
    try: #python3
        from urllib.request import urlopen
    except: #python2
        from urllib2 import urlopen

    from os.path import split, join

    curdir = split(__file__)[0]
    print(url)
    

    if dest_fname is None:
        dest_fname = join(curdir, split(url)[1])

    try:
        contents = urlopen(url).read()
    except:
        raise Exception('Unable to get url: %s' % (url,))
    open(dest_fname, 'wb').write(contents)
