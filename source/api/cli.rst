=================
CLI Extension API
=================

.. module:: secureml.cli

The SecureML CLI is built using the `Click <https://click.palletsprojects.com/>`_ library, which makes it straightforward to extend with custom commands. This reference explains how to programmatically extend the CLI with new functionality.

CLI Architecture
---------------

The SecureML CLI follows a nested command structure:

.. code-block:: text

    secureml                   # Root command group
    ├── anonymization          # Command group for anonymization features
    │   └── k-anonymize        # Command for k-anonymity
    ├── compliance             # Command group for compliance features
    │   └── check              # Command for compliance checking
    ├── synthetic              # Command group for synthetic data features
    │   └── generate           # Command for data generation
    ├── presets                # Command group for regulation presets
    │   ├── list               # Command to list presets
    │   └── show               # Command to show preset details
    ├── environments           # Command group for isolated environments
    │   ├── setup-tf-privacy   # Command to setup TF Privacy environment
    │   └── info               # Command to show environment info
    └── keys                   # Command group for key management
        ├── configure-vault    # Command to configure Vault
        ├── generate-key       # Command to generate a key
        └── get-key            # Command to retrieve a key

Adding a New Command Group
-------------------------

To add a new command group to the CLI, use the `@cli.group()` decorator:

.. code-block:: python

    from secureml.cli import cli
    
    @cli.group()
    def my_extension():
        """Description of your extension functionality."""
        pass
    
    @my_extension.command()
    @click.argument("input_file", type=click.Path(exists=True))
    @click.option("--option", "-o", help="Description of the option")
    def my_command(input_file, option):
        """Detailed description of what your command does.
        
        This will appear in the help text.
        """
        # Your command implementation here
        click.echo(f"Processing {input_file} with option {option}")

Adding a Command to an Existing Group
------------------------------------

You can extend existing command groups by importing them:

.. code-block:: python

    from secureml.cli import anonymization
    
    @anonymization.command("my-anonymizer")
    @click.argument("input_file", type=click.Path(exists=True))
    @click.argument("output_file", type=click.Path())
    def my_anonymizer(input_file, output_file):
        """Custom anonymization implementation."""
        # Your command implementation here
        click.echo(f"Anonymizing {input_file} to {output_file}")

Command Arguments and Options
----------------------------

SecureML CLI commands use Click's argument and option decorators:

.. code-block:: python

    @click.argument("name")                     # Required positional argument
    @click.argument("path", type=click.Path())  # Path argument with validation
    @click.option("--verbose", is_flag=True)    # Boolean flag option
    @click.option("--count", type=int)          # Option with type validation
    @click.option(
        "--method",                          
        type=click.Choice(["a", "b", "c"]),     # Option with choices
        default="a",
    )
    @click.option("--items", "-i", multiple=True)  # Option that can be repeated

Integration with SecureML Features
---------------------------------

When extending the CLI, you can use SecureML's core functionality:

.. code-block:: python

    from secureml import anonymize
    from secureml.cli import cli
    
    @cli.group()
    def custom():
        """Custom operations."""
        pass
    
    @custom.command("protect-data")
    @click.argument("input_file", type=click.Path(exists=True))
    @click.argument("output_file", type=click.Path())
    @click.option("--method", default="k-anonymity", help="Anonymization method")
    def protect_data(input_file, output_file, method):
        """Process data with custom protection rules."""
        import pandas as pd
        
        # Load data
        data = pd.read_csv(input_file)
        
        # Use SecureML's anonymization
        result = anonymize(
            data=data,
            method=method,
            sensitive_columns=["name", "email"]
        )
        
        # Save result
        result.to_csv(output_file, index=False)
        click.echo(f"Protected data saved to {output_file}")

Packaging Your Extension
-----------------------

Create a plugin package that automatically extends the CLI when installed:

1. Create a Python package with your extension
2. In your package's `setup.py`, include an entry point:

.. code-block:: python

    setup(
        name="secureml-myext",
        version="0.1.0",
        packages=find_packages(),
        install_requires=["secureml"],
        entry_points="""
            [secureml.cli_plugins]
            myext=secureml_myext.cli:register_commands
        """,
    )

3. In your package, implement the function that registers your commands:

.. code-block:: python

    # secureml_myext/cli.py
    from secureml.cli import cli
    
    def register_commands():
        """Register custom commands with SecureML CLI."""
        @cli.group()
        def myext():
            """My SecureML extension commands."""
            pass
        
        @myext.command("process")
        @click.argument("input_file")
        def process(input_file):
            """Process files with my extension."""
            click.echo(f"Processing {input_file} with my extension")

Best Practices
-------------

1. **Follow the command pattern**: Keep your commands consistent with SecureML's style
2. **Provide comprehensive help**: Document your commands thoroughly
3. **Use proper types**: Validate arguments and options with Click's type system
4. **Handle errors gracefully**: Use Click's error handling features
5. **Progress indicators**: For long-running operations, use Click's progress features:

.. code-block:: python

    with click.progressbar(range(100), label="Processing") as bar:
        for i in bar:
            # Do work here
            pass

6. **Color and formatting**: Use Click's styling to make your output more readable:

.. code-block:: python

    click.secho("Success!", fg="green", bold=True)
    click.secho("Error:", fg="red", nl=False)
    click.echo(" Something went wrong")

7. **Confirm destructive operations**:

.. code-block:: python

    if click.confirm("This will overwrite existing data. Continue?"):
        # Proceed with operation
        pass 