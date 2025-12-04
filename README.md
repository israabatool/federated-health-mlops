# Federated Health MLOps Project 
 
This repository contains an end-to-end federated learning & MLOps system for health-risk prediction. 
Federated Health Analytics MLOps System – README

A privacy-preserving, multi-node Federated Learning (FL) + MLOps pipeline for real-time health-risk prediction using wearable, IoT, and weather data.

Project Overview

This project builds a Federated Learning system where hospitals (nodes) collaboratively train a machine learning model without sharing raw data. Only model updates are exchanged, ensuring privacy and compliance.

The pipeline integrates:

TensorFlow Federated (TFF)

Docker + Kubernetes

MLflow

Prometheus + Grafana

Streamlit dashboards

This README documents all progress from Phase 1 to Phase 7.

PHASE 1 — Project Initialization & Planning (Completed)
Requirements Defined

Build a multi-source, privacy-preserving health risk prediction system.

Implement Federated Learning + MLOps end-to-end.

Three hospital nodes:

Node A – Hospital A

Node B – Hospital B

Node C – Hospital C

Tools Selected
Component	Tool
Federated Learning	TensorFlow Federated (TFF)
Deployment	Docker, Kubernetes
Tracking	MLflow
Monitoring	Prometheus, Grafana
Dashboards	Streamlit
Data Sources

Wearables → heart_rate, spo2, steps

IoT Air Quality → pm25, co2

Weather API → temperature, humidity

Target → risk label (0/1)

Folder Structure
/data
/federated_training
/mlops
/dashboard
/k8s
/outputs

Deliverables

Requirements document

Architecture diagram

Tools setup

Initial GitHub repository structure

PHASE 2 — Data Ingestion & Simulation (Completed)
Node-wise datasets created

node_A_data.csv

node_B_data.csv

node_C_data.csv

Each dataset contains:

heart_rate | spo2 | steps | pm25 | co2 | temp | humidity | risk

EDA Completed

Missing value analysis

Correlation heatmaps

Distribution plots

Outlier detection

Feature summary

Preprocessing Implemented

Scripts handle:

Normalization

Missing value cleaning

Outlier correction

Timestamp processing

Deliverables

Clean CSVs

preprocessing.py

EDA notebook

Data dictionary

PHASE 3 — Federated Data Pipeline (Completed)
TFF ClientData Created

Converted Pandas → tf.data.Dataset

Consistent schema across all nodes

Client sampling verified

Tests Verified

All client IDs load successfully

Feature shapes and dtypes match

No schema mismatch

Deliverables

make_tff_data.py

Verified federated client dataset

Outcome:
Each hospital acts as an independent TFF client ready for federated training.

PHASE 4 — Model Design (Completed)
Model Architecture

A compact neural network compatible with TensorFlow Federated:

Input → Dense(64) → Dense(32) → Dense(1, activation='sigmoid')

Hyperparameters

Loss: Binary Crossentropy

Client Optimizer: Adam

Server Optimizer: SGD or Adam

Metrics: Accuracy, Recall, AUC

Deliverables

model_def.py

Architecture diagram

Hyperparameter documentation

PHASE 5 — Federated Training (Completed)
Implemented Federated Averaging (FedAvg)

Wrapped Keras model into a TFF model

Client update logic implemented

Server aggregation defined

Multi-round training loop

Per-round metric logging

Outputs Generated

Training logs

Loss and accuracy per round

Global model weights (final_fed_model_weights.pkl)

Deliverables

tff_train.py

Training logs

Global model file

Outcome:
The system trained a unified global model while keeping all node-level data private.

PHASE 6 — Evaluation & Benchmarking (Completed)
Model evaluated on:

Node A test set

Node B test set

Node C test set

Combined validation dataset

Metrics Calculated

Accuracy

Precision

Recall

F1-score

AUC

Confusion Matrix

ROC Curve

Comparisons Performed

Federated model vs. centralized model

Per-node performance differences

Deliverables

evaluate.py

Evaluation report

Visualizations (ROC curves, confusion matrices, etc.)

PHASE 7 — Data Drift Detection (Completed)
Drift Detection Techniques

Kolmogorov–Smirnov (KS) test

Mean/variance shift detection

Seasonal pattern analysis

Population changes and drift in distributions

Drift Visualization Dashboard

Includes:

Sensor anomalies

Climate-variable drift

Wearable device irregularities

Automatic drift alerts

Deliverables

drift_detection.py

Drift dashboard

Automated alert system
