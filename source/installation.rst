============
Installation
============

SecureML can be installed using pip or poetry, with various optional components available based on your needs.

Requirements
-----------

* Python 3.11
* Operating Systems: Linux, macOS, Windows

Basic Installation
----------------

The simplest way to install SecureML is using pip:

.. code-block:: bash

    pip install secureml

Using Poetry
-----------

For development or if you're using Poetry for dependency management:

.. code-block:: bash

    poetry add secureml

Optional Components
-----------------

SecureML offers several optional components that can be installed based on your needs:

PDF Report Generation
^^^^^^^^^^^^^^^^^^^^

For generating PDF reports for compliance and audit trails with WeasyPrint:

.. code-block:: bash

    pip install secureml[pdf]

On Windows, WeasyPrint requires GTK libraries. See `installation guide <https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows>`_.

HashiCorp Vault Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^

For secure key management with HashiCorp Vault:

.. code-block:: bash

    pip install secureml[vault]

All Optional Components
^^^^^^^^^^^^^^^^^^^^^

To install all optional components:

.. code-block:: bash

    pip install secureml[pdf,vault]

Isolated Environments
-------------------

Some components like TensorFlow Privacy are installed in isolated environments to prevent dependency conflicts. 
SecureML uses this approach to handle dependencies that would otherwise conflict with the main package.

When you use functionality requiring TensorFlow Privacy, SecureML will:

1. Automatically create a separate virtual environment (first time only)
2. Install the required dependencies in that environment
3. Execute the relevant code there and return results to your main environment

You can pre-configure these environments using the CLI:

.. code-block:: bash

    # Set up the TensorFlow Privacy environment
    secureml environments setup-tf-privacy

For more detailed information, see the :doc:`isolated_environments` section of the user guide.

Development Installation
----------------------

For development purposes, clone the repository and install in development mode:

.. code-block:: bash

    git clone https://github.com/scimorph/secureml.git
    cd secureml
    poetry install

Verifying Installation
--------------------

You can verify your installation by running:

.. code-block:: python

    import secureml
    print(secureml.__version__)

Or using the CLI:

.. code-block:: bash

    secureml --version 