PySensors
=========
|Build| |PyPI|

**PySensors** is a Python package for sparse sensor placement.


Installation
-------------

Installing with pip
^^^^^^^^^^^^^^^^^^^

If you are using Linux or macOS you can install PySensors with pip (note the name you type in here is slightly different "pysensors"):

.. code-block:: bash

  pip install python-sensors

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
First clone this repository:

.. code-block:: bash

  git clone https://github.com/dynamicslab/pysensors.git

Then, to install the package, run

.. code-block:: bash

  pip install .

If you do not have pip you can instead use

.. code-block:: bash

  python setup.py install

If you do not have root access, you should add the ``--user`` option to the above lines.


Community guidelines
--------------------

Contributing code
^^^^^^^^^^^^^^^^^
We welcome contributions to PySensors. To contribute a new feature please submit a pull request. To get started we recommend installing the packages in ``requirements-dev.txt`` via

.. code-block:: bash

    pip install -r requirements-dev.txt

This will allow you to run unit tests and automatically format your code. To be accepted your code should conform to PEP8 and pass all unit tests. Code can be tested by invoking

.. code-block:: bash

    pytest

We recommed using ``pre-commit`` to format your code. Once you have staged changes to commit

.. code-block:: bash

    git add path/to/changed/file.py

you can run the following to automatically reformat your staged code

.. code-block:: bash

    pre-commit -a -v

Note that you will then need to re-stage any changes `pre-commit` made to your code.

Reporting issues or bugs
^^^^^^^^^^^^^^^^^^^^^^^^
If you find a bug in the code or want to request a new feature, please open an issue.

References
------------
-  Manohar, Krithika, Bingni W. Brunton, J. Nathan Kutz, and Steven L. Brunton.
   "Data-driven sparse sensor placement for reconstruction: Demonstrating the
   benefits of exploiting known patterns."
   IEEE Control Systems Magazine 38, no. 3 (2018): 63-86.
   `[DOI] <10.1109/MCS.2018.2810460>`__

-  Clark, Emily, Travis Askham, Steven L. Brunton, and J. Nathan Kutz.
   "Greedy sensor placement with cost constraints." IEEE Sensors Journal 19, no. 7
   (2018): 2642-2656.
   `[DOI] <10.1109/JSEN.2018.2887044>`__
   
.. |Build| image:: https://github.com/dynamicslab/pysensors/workflows/Tests/badge.svg
    :target: https://github.com/dynamicslab/pysensors/actions?query=workflow%3ATests

.. |PyPI| image:: https://badge.fury.io/py/python-sensors.svg
    :target: https://badge.fury.io/py/python-sensors
