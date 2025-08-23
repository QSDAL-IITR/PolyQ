Contribution Guide
==================

Thank you for considering contributing to PolyQ!  

Getting started
---------------

1. Fork the repo and clone your fork.
2. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

3. Make your changes and commit them:

   .. code-block:: bash

      git commit -m "Add your commit message here"

4. Push your changes to your fork:

   .. code-block:: bash

      git push origin feature/your-feature-name

5. Open a pull request against the ``main`` branch of the original repository.
6. Ensure your pull request passes the CI checks.
7. Wait for review and address any feedback.
8. Once approved, your changes will be merged into the main branch.

Contributing to Documentation
=============================

The project uses Sphinx for documentation generation. Here's how to contribute:

Setting up documentation environment
------------------------------------

1. Install documentation dependencies:

   .. code-block:: bash

      pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme

2. Install project dependencies (required for autodoc):

   .. code-block:: bash

      pip install numpy qiskit psutil

Files to modify
---------------

* **API Documentation**: Files in ``PolyQ/`` - Add docstrings to functions and classes
* **User Guide**: ``docs/source/index.rst`` - Main documentation page
* **Module References**: ``docs/source/PolyQ.rst`` - Auto-generated API reference
* **Contributing Guide**: ``docs/source/contributing.rst`` - This file
* **Configuration**: ``docs/source/conf.py`` - Sphinx configuration

Building documentation
----------------------

1. Navigate to the docs directory:

   .. code-block:: bash

      cd docs

2. Build the HTML documentation:

   .. code-block:: bash

      make html

3. View the generated documentation:

   .. code-block:: bash

      open _build/html/index.html

4. For clean builds (recommended after major changes):

   .. code-block:: bash

      make clean
      make html

Documentation style guidelines
------------------------------

* Use clear, concise language
* Include examples in docstrings where helpful
* Follow Google-style docstring format:

   .. code-block:: python

      def example_function(param1: str, param2: int) -> bool:
          """Brief description of the function.
          
          Args:
              param1: Description of parameter 1
              param2: Description of parameter 2
              
          Returns:
              Description of return value
              
          Example:
              >>> example_function("hello", 5)
              True
          """