def fix_warnings():
    import platform
    if platform.system() == 'Darwin':
        # Get rid of various flags that cause warnings and/or errors
        import distutils.sysconfig as ds
        import string
        a = ds.get_config_vars()
        for k, v in list(a.items()):
            try:
                if string.find(v, '-Wstrict-prototypes') != -1:
                    a[k] = string.replace(v, '-Wstrict-prototypes', '')
                if string.find(v, '-arch i386') != -1:
                    a[k] = string.replace(v, '-arch i386', '')
                if string.find(v, '-mno-fused-madd ') != -1:
                    a[k] = string.replace(v, '-mno-fused-madd', '')
            except:
                pass