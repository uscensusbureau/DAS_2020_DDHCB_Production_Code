
.PHONY: assemble package release clean upload

#Default to what's on the path unless set in an env var
PIP3 ?= pip3
PYTHON3 ?= python3

assemble: clean
	mkdir -p target
	cp setup.py ../safetab_h/

package: assemble prereqs
	cd ../analytics && $(PYTHON3) -m build -n 
	cd ../common && $(PYTHON3) -m build -n 
	cd ../safetab_h && $(PYTHON3) -m build -n
	cd ../safetab_utils && $(PYTHON3) -m build -n 
	cp ../*/dist/* target 
	cp ../core/*.whl target

prereqs:
	$(PIP3) install --upgrade pip build wheel setuptools pkginfo


install: package 
	$(PIP3) install --force-reinstall target/*.whl


upload: package
	$(PIP3) install --upgrade twine
	$(PYTHON3) -m twine upload --verbose --skip-existing --repository-url ${NEXUS_REPO} target/*


clean:
	rm -rf target 
	rm -rf ../*/dist
	rm -f ../safetab_h/setup.py
