==========
User Guide
==========

This user guide provides detailed information on using SecureML's features to create privacy-preserving machine learning systems.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   anonymization
   differential_privacy
   synthetic_data
   compliance
   federated_learning
   audit_trails
   key_management
   cli
   isolated_environments

Overview
--------

SecureML is designed to help machine learning engineers build privacy-preserving AI systems that comply with regulations like GDPR, CCPA, and HIPAA. The library provides tools across the entire machine learning lifecycle:

Data Preparation
^^^^^^^^^^^^^^^

* **Anonymization**: Transform sensitive data to protect individual privacy while maintaining utility
* **Synthetic Data**: Generate realistic but artificial data that preserves statistical properties

Model Training
^^^^^^^^^^^^^

* **Differential Privacy**: Train models with mathematical privacy guarantees
* **Federated Learning**: Train models across decentralized data sources without sharing raw data

Compliance & Auditing
^^^^^^^^^^^^^^^^^^^^

* **Compliance Checking**: Verify that your datasets and models comply with privacy regulations
* **Audit Trails**: Maintain comprehensive logs of all data operations for compliance documentation

Security
^^^^^^^

* **Key Management**: Securely store and manage encryption keys using HashiCorp Vault
* **Command Line Interface**: Perform secure operations via a comprehensive CLI
* **Isolated Environments**: Manage dependency conflicts with advanced environment isolation 