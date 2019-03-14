#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['dummy']

class Dummy(object):
    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            pass
        return wrapper

dummy = Dummy()


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs