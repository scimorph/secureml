====================
Isolated Environments
====================

Overview
--------

SecureML uses isolated virtual environments to manage dependencies that have conflicting requirements with the main package. This architectural approach allows SecureML to seamlessly integrate with packages like TensorFlow Privacy, which requires specific package versions that would otherwise conflict with SecureML's core dependencies.

.. note::
   Currently, the primary use case for isolated environments is TensorFlow Privacy, which requires ``packaging ~= 22.0``, while other SecureML dependencies require ``packaging >= 24.0``.

Why Isolated Environments?
-------------------------

Machine learning libraries often have complex dependency trees with specific version requirements. In particular:

- **Version Conflicts**: Libraries like TensorFlow Privacy may require versions of dependencies that conflict with other parts of SecureML
- **Dependency Bloat**: Installing all possible dependencies would make the package unnecessarily large
- **User Experience**: We want to provide a seamless experience without forcing users to manage complex environments manually

How Isolated Environments Work
-----------------------------

When you use functionality that requires TensorFlow Privacy through SecureML, the library:

1. **Automatic Management**: Automatically creates and manages a separate Python virtual environment
2. **Transparent Integration**: Handles all communication between your main environment and the isolated environment
3. **Efficient Resource Usage**: Only creates the environment when needed

The Architecture
^^^^^^^^^^^^^^^

.. code-block:: text

    Main Python Environment                 Isolated Environment
    ┌───────────────────────────┐          ┌──────────────────────────┐
    │                           │          │                          │
    │  Your Application         │          │  TensorFlow Privacy      │
    │  ┌──────────────────┐     │          │  Environment             │
    │  │                  │     │  JSON    │                          │
    │  │  SecureML        │─────┼─────────▶│  • TensorFlow            │
    │  │                  │     │  IPC     │  • TensorFlow Privacy    │
    │  └──────────────────┘     │          │  • Numpy, Pandas         │
    │                           │          │  • SecureML              │
    └───────────────────────────┘          └──────────────────────────┘

- **Communication**: SecureML uses a secure JSON-based communication protocol to transfer data between environments
- **Serialization**: Model parameters, datasets, and results are serialized when passing between environments
- **Error Handling**: Any errors in the isolated environment are properly captured and reported back to the main environment

TensorFlow Privacy Integration
-----------------------------

The most common use case for isolated environments is when using SecureML's differential privacy functionality with TensorFlow.

Default Behavior
^^^^^^^^^^^^^^^

By default, when you call ``differentially_private_train()`` with ``framework="tensorflow"``, SecureML will:

1. Check if the TensorFlow Privacy environment exists
2. Create it if it doesn't exist (this happens only once)
3. Send your model and data to the isolated environment
4. Run the training in the isolated environment
5. Return the trained model back to your main environment

Example Usage
^^^^^^^^^^^^

.. code-block:: python

    from secureml import privacy
    import tensorflow as tf
    
    # Create a model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train with differential privacy
    private_model = privacy.differentially_private_train(
        model=model,
        data=training_data,
        epsilon=1.0,
        delta=1e-5,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        framework="tensorflow"  # This triggers the isolated environment
    )

    # The model is trained with differential privacy guarantees and returned to your main environment
    predictions = private_model.predict(test_data)

Managing Isolated Environments
-----------------------------

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^

SecureML provides CLI commands to manage isolated environments:

.. code-block:: bash

    # Set up the TensorFlow Privacy environment in advance
    secureml environments setup-tf-privacy

    # Force recreation of the environment (useful for troubleshooting)
    secureml environments setup-tf-privacy --force

    # Check the status of isolated environments
    secureml environments info

Using the API
^^^^^^^^^^^^

You can also manage isolated environments programmatically:

.. code-block:: python

    from secureml.isolated_environments import (
        setup_tf_privacy_environment, 
        is_env_valid,
        get_env_path
    )
    
    # Set up the environment
    setup_tf_privacy_environment()
    
    # Check if the environment is valid
    if is_env_valid():
        print("Environment is ready for use")
    else:
        print("Environment needs to be set up")
    
    # Get the path to the environment
    env_path = get_env_path()
    print(f"TensorFlow Privacy environment is at: {env_path}")

Location and Structure
^^^^^^^^^^^^^^^^^^^^^

By default, isolated environments are created at:

- **Linux/macOS**: ``~/.secureml/tf_privacy_venv``
- **Windows**: ``%USERPROFILE%\.secureml\tf_privacy_venv``

The environment contains:

- Python interpreter
- TensorFlow (compatible version)
- TensorFlow Privacy
- NumPy and Pandas
- A copy of SecureML

Advanced Topics
--------------

Custom Environment Path
^^^^^^^^^^^^^^^^^^^^^^

Currently, SecureML does not support customizing the environment path, but this feature is planned for future releases.

Troubleshooting
^^^^^^^^^^^^^^

If you encounter issues with the isolated environment:

1. **Recreate the environment**:

   .. code-block:: bash

       secureml environments setup-tf-privacy --force

2. **Check for errors during setup**:

   .. code-block:: bash

       secureml environments setup-tf-privacy --verbose

3. **Verify installed packages**:

   .. code-block:: bash

       # Linux/macOS
       ~/.secureml/tf_privacy_venv/bin/pip list
       
       # Windows
       %USERPROFILE%\.secureml\tf_privacy_venv\Scripts\pip list

4. **Manual cleanup** (if necessary):

   .. code-block:: bash

       # Remove the environment directory
       rm -rf ~/.secureml/tf_privacy_venv  # Linux/macOS
       rmdir /s /q %USERPROFILE%\.secureml\tf_privacy_venv  # Windows
       
       # Then recreate it
       secureml environments setup-tf-privacy

Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^

- **First-time setup**: The first time you use TensorFlow Privacy functionality, there will be a delay as the environment is created and packages are installed
- **Subsequent usage**: After the initial setup, the overhead is minimal, primarily related to data serialization/deserialization
- **Memory usage**: The isolated environment runs in a separate process, which requires additional memory

Implementation Details
--------------------

For developers interested in how isolated environments are implemented:

- The `run_tf_privacy_function()` function manages the execution of code in the isolated environment
- Communication happens through temporary files containing JSON-serialized data
- A subprocess is created to run Python code in the isolated environment
- The result is returned through another temporary file

Future Plans
^^^^^^^^^^^

In future releases, we plan to:

- Support custom environment locations
- Add more isolated environments for other conflicting dependencies
- Improve error reporting and logging
- Add support for memory-mapped communication for better performance with large datasets 