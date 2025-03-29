# Smart Traffic Management System

A smart AI-based solution for traffic management on routes with heavy traffic from different directions, with real-time monitoring and adaptation of traffic light timings.

## Features

- Real-time vehicle detection using computer vision
- Traffic density analysis by direction
- AI-based traffic prediction using LSTM neural networks
- Adaptive traffic light timing optimization
- Simulation environment for testing
- User interface for monitoring and manual control
- REST API for integration with other systems

## Project Structure

smart_traffic_management/ ├── data/ │ ├── training_data/ │ └── test_data/ ├── models/ │ ├── saved_models/ │ └── model_checkpoints/ ├── src/ │ ├── monitoring/ │ │ ├── camera.py │ │ ├── vehicle_detection.py │ │ └── traffic_density.py │ ├── processing/ │ │ ├── data_preprocessing.py │