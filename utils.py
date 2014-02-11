"""
Copyright (C) 2013
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""


import urllib2
from os.path import split, join


def wget(url, dest_fname=None):
    curdir = split(__file__)[0]
    if dest_fname is None:
        dest_fname = join(curdir, split(url)[1])

    try:
        contents = urllib2.urlopen(url).read()
    except:
        raise Exception('Unable to get url: %s' % (url,))
    open(dest_fname, 'w').write(contents)
