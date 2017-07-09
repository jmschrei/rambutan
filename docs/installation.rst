.. _installation:

Installation
============

Installation of Rambutan is complicated by two of its dependencies-- mxnet and cython. If you have these two packages installed, then all you need to do is download the current code with the following commands:

.. code-block:: bash

	git clone https://github.com/jmschrei/rambutan
	cd rambutan
	python setup.py install

It is difficult to install either mxnet or cython on a windows machine. mxnet has a great deal of dependencies that can dramatically influence the speed of Rambutan and should not be ignored. In particular, cuDNN is very valuable when it comes to accelerating the models. 

mxnet installation instructions can be found at http://mxnet.io/get_started/install.html

cython requires a working c++ compiler. Both Mac (clang) and Linux (gcc) machines come with working compilers. For Python 2, this minimal version of Visual Studio 2008 has worked well in the past: https://www.microsoft.com/en-us/download/details.aspx?id=44266. For Python 3, this version of the Visual Studio Build Tools has worked in the past: http://go.microsoft.com/fwlink/?LinkId=691126. If neither of those work, then https://wiki.python.org/moin/WindowsCompilers may provide more information. Note that your compiler version must match your python version. Run `python --version` to tell which python version you use. Don't forget to select the appropriate Windows version API you'd like to use. If you get an error message "ValueError: Unknown MS Compiler version 1900" remove your Python's Lib/distutils/distutil.cfg and retry. See http://stackoverflow.com/questions/34135280/valueerror-unknown-ms-compiler-version-1900 for details. 

It is suggested that one start out with the Anaconda or Enthought python distributions that come with all other dependencies pre-installed. This will significant speed up the starting time for those who are getting started with Python.
