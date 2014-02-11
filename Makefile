
all: contexts/ctx_mac.c contexts/ctx_mesa.c
	python2.7 setup.py build_ext --inplace

contexts/ctx_mac.c:
	make -C contexts

contexts/ctx_mesa.c:
	make -C contexts

sdist: all
	python setup.py sdist && rsync -avz dist/drender-0.5.tar.gz files:~/drender/latest.tgz

clean:
	rm -rf `find . -name \*.pyc` `find . -name \*~` build/ dist/; make -C contexts clean

realclean: clean
	rm -rf contexts/OSMesa; rm -rf contexts/OSMesa.*.zip

test: all
	env LD_PRELOAD=$(PRELOADED) python -m unittest discover


