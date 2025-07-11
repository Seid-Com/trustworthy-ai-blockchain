# Data Availability Statement

## Overview

This repository contains a comprehensive implementation of blockchain-integrated federated learning systems for trustworthy AI research. All data used in this implementation is synthetically generated for educational and research purposes.

## Data Sources

### Synthetic Data Generation

The application generates synthetic datasets to demonstrate federated learning concepts:

- **MNIST-like Image Data**: 28x28 grayscale images with 10 classes
- **Client Data Distribution**: Non-IID data across federated clients
- **Blockchain Transactions**: Simulated model updates and data entries
- **Attack Scenarios**: Realistic attack patterns and detection metrics

### Data Generation Functions

Located in `utils.py`:

```python
def generate_mnist_data(num_samples: int = 1000, client_id: str = "client_0")
```

- Generates synthetic MNIST-like datasets
- Creates non-IID distribution across clients
- Ensures reproducible results with seeded random generation

### Blockchain Data

The blockchain simulator (`blockchain_simulator.py`) generates:

- **Block Data**: Timestamps, hashes, Merkle roots
- **Transaction Records**: Model updates, data entries, consensus votes
- **Provenance Tracking**: Complete audit trail of all operations

### Performance Metrics

The metrics tracker (`metrics_tracker.py`) collects:

- **Training Metrics**: Accuracy, loss, convergence rates
- **Security Metrics**: Attack detection rates, false positives
- **System Metrics**: Communication overhead, energy consumption
- **Blockchain Metrics**: Block creation time, validation efficiency

## Reproducibility

### Deterministic Generation

All synthetic data generation uses seeded random number generators:

- **NumPy random seed**: Set per client for consistent data distribution
- **Sklearn random state**: Fixed for reproducible model training
- **Simulation parameters**: Configurable through the interface

### Configuration Parameters

Default system parameters for reproducible results:

```python
DEFAULT_CLIENTS = 10
DEFAULT_VALIDATORS = 10
DEFAULT_TRAINING_ROUNDS = 20
DEFAULT_BATCH_SIZE = 32
BYZANTINE_TOLERANCE = 0.33
```

## Data Access

### Local Access

All data is generated in real-time within the application:

1. **Client Data**: Generated when federated clients are initialized
2. **Blockchain Data**: Created during simulation runs
3. **Metrics Data**: Collected during training and consensus rounds
4. **Attack Data**: Generated during security simulations

### Export Capabilities

The application provides data export functions:

- **CSV Export**: Performance metrics and training history
- **JSON Export**: Blockchain data and transaction records
- **Visualization Data**: Charts and graphs for analysis

### Session Persistence

Data is maintained in Streamlit session state:

- **Training History**: All rounds and client updates
- **Blockchain State**: Complete chain with all blocks
- **Metrics Timeline**: Historical performance data
- **Attack Logs**: Security event records

## Research Applications

### Academic Use

This implementation is designed for:

- **Educational Demonstrations**: Teaching blockchain and federated learning concepts
- **Research Prototyping**: Testing new algorithms and approaches
- **Comparative Studies**: Baseline vs blockchain-integrated systems
- **Security Analysis**: Attack simulation and defense mechanisms

### Benchmarking

The system provides baseline comparisons:

- **Standard Federated Learning**: Traditional centralized aggregation
- **Blockchain-Integrated**: Enhanced with consensus and provenance
- **Performance Metrics**: Detailed comparison tables and visualizations
- **Security Metrics**: Attack success rates and detection efficiency

## Technical Implementation

### Data Structures

Key data structures used throughout the system:

```python
# Client data structure
{
    "client_id": str,
    "data_size": int,
    "model_accuracy": float,
    "training_rounds": int,
    "is_malicious": bool
}

# Blockchain block structure
{
    "index": int,
    "timestamp": str,
    "data": List[Dict],
    "previous_hash": str,
    "hash": str,
    "merkle_root": str,
    "nonce": int
}

# Training metrics structure
{
    "round": int,
    "global_accuracy": float,
    "client_accuracies": List[float],
    "communication_cost": float,
    "consensus_time": float,
    "attacks_detected": int
}
```

### Storage Format

Data is stored in memory using Python dictionaries and lists:

- **JSON-serializable**: All data structures can be exported
- **Numpy arrays**: Model weights and training data
- **Pandas DataFrames**: Metrics and performance data
- **NetworkX graphs**: Validator networks and topology

## Validation

### Data Integrity

The system validates data integrity through:

- **Cryptographic Hashing**: SHA-256 for all data entries
- **Merkle Trees**: Efficient verification of data batches
- **Digital Signatures**: Simulated client authentication
- **Consensus Validation**: PBFT agreement on data validity

### Quality Assurance

Quality checks implemented:

- **Data Distribution**: Verification of non-IID properties
- **Model Convergence**: Training progress validation
- **Attack Simulation**: Realistic attack patterns
- **Performance Baselines**: Comparison with literature benchmarks

## Compliance

### Research Ethics

This implementation follows research ethics guidelines:

- **No Personal Data**: All data is synthetically generated
- **Privacy Preservation**: Demonstrates privacy-preserving techniques
- **Transparency**: Complete source code availability
- **Reproducibility**: Deterministic generation and clear documentation

### Open Science

Supporting open science principles:

- **Open Source**: MIT license for broad accessibility
- **Documentation**: Comprehensive implementation details
- **Reproducible Research**: Clear methodology and parameters
- **Community Contribution**: GitHub repository for collaboration

## Contact

For questions about data availability or technical implementation:

- **Repository**: https://github.com/Seid-Com/trustworthy-ai-blockchain
- **Issues**: GitHub issue tracker for technical questions
- **Documentation**: README.md and inline code comments
- **Citation**: BibTeX entry provided in README.md

## Future Enhancements

Planned improvements for data availability:

- **Real Dataset Integration**: Support for actual federated learning datasets
- **Database Backend**: Persistent storage for large-scale experiments
- **API Endpoints**: RESTful access to generated data
- **Visualization Export**: Enhanced chart and graph export capabilities