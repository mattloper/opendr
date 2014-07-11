
all: 
	python2.7 setup.py build_ext --inplace

clean:
	rm -rf `find . -name \*.pyc` `find . -name \*~` build/ *.so _constants.py _functions.pyx ctx_mac.c ctx_mesa.c

realclean: clean
	rm -rf OSMesa; rm -rf OSMesa.*.zip

test: all
	env LD_PRELOAD=$(PRELOADED) python -m unittest discover


