# Software Requirements Analysis Specification Document

| Date         | Version | Description                                                  |
| ------------ | ------- | ------------------------------------------------------------ |
| <12/31/2020> | <1.0>   | First Draft of the Software Requirement Analysis Specification Doc. |

[toc]

## 1 Preamble

The Software Requirements Analysis Specification is designed to document and describe the agreement between the software user and the developer regarding the specification of the software product requested. Its primary purpose is to provide a clear and descriptive “statement of user requirements” that can be used as a reference in the further development of the software system. This document is broken into a number of sections used to logically separate the software requirements into easily referenced parts.

This Software Requirements Analysis Specification aims to describe the functionality, external interfaces, attributes and design constraints imposed on the implementation of the software system described throughout the rest of the document. Throughout the description of the software system, the language and terminology used should be unambiguous and consistent throughout the document.

### 1.1 Purpose

The purpose of this project is to study and investigate the method of DAD, for example TSI+CNN, RNN, LSTM, etc. Based on above study and survey, it is expected to **provide a program with simple, user-friendly front-end display page for anomaly detection based on time series data**, which can read the file uploaded by users and support users to select specific models for anomaly detection.

The goal of this project is to apply and practice software engineering methods and ideas during the scientific research process, from requirement analysis to modeling, to architecture design and detailed design, to the final code implementation and review at each stage. Throughout the project cycle, we always adopt the agile development model, hold weekly meetings, report on the work progress of the previous week, discuss solutions to problems encountered in the previous stage, and plan and assign tasks for the next week. At the same time, in order to make the whole project clear, we learned and tried many process management tools and project management tools, such as Microsoft Todo, github, postman, curtain, etc.

### 1.2 Background

#### 1.2.1 Anomaly

A common need when analyzing real-world data-sets is determining which instances stand out as being dissimilar to all others. Such instances are known as *anomalies*, and the goal of *anomaly detection* (also known as *outlier* *detection*) is to determine all such instances in a data-driven fashion (Chandola et al. [2007]). Anomalies can be caused by errors in the data but sometimes are indicative of a new, previously unknown, underlying process; Hawkins [1980] defines an outlier as an observation that *deviates so significantly from other observations as to arouse suspicion* *that it was generated by a different mechanism.* 

#### 1.2.2 Deep learning

In the broader field of machine learning, the recent years have witnessed a proliferation of deep neural networks, with unprecedented results across various application domains. Deep learning is a subset of machine learning that achieves good performance and flexibility by learning to represent the data as a nested hierarchy of concepts within layers of the neural network. Deep learning outperforms the traditional machine learning as the scale of data increases. In recent years, deep learning-based anomaly detection algorithms have become increasingly popular and have been applied for a diverse set of tasks; studies have shown that deep learning completely surpasses traditional methods (Javaid et al. [2016], Peng and Marculescu [2015]). 

#### 1.2.3 Deep Anomaly Detection(DAD)

This project focuses on the research and program development of DAD（Deep Anomoly Detection）drawing on expertise from School of Software Engineering. DAD firstly uses some transforms to modify the original time series data , then uses Deep learning to classify the normal and abnormal points. 

### 1.3 Definitions, Acronyms and Abbreviations

**Table of Glossary**

| Terms                | Definition                                                   |
| :------------------- | :----------------------------------------------------------- |
| Machine Learning(ML) | It is specialized in studying how computers simulate or implement human learning behavior to acquire new knowledge or skills, and reorganize existing knowledge structure to continuously improve their own performance. |
| Deep Learning(DL)    | A new research direction in the field of Machine Learning    |
| DAD                  | Deep Anomaly Detection                                       |
| TSI                  | Time Series to Image                                         |
| FNN                  | Feedforward Neural Networks                                  |
| CNN                  | Convolutional Neural Networks<br />a kind of FNN with deep structure including convolution computation is one of the representative algorithms of deep learning |
| RNN                  | Recurrent Neural Network<br />a kind of recursive neural network which takes sequence data as input, recurses in the evolution direction of the sequence and connects all nodes (cycle units) in a chain way |
| LSTM                 | Long Short-Term Memory<br />a time cycle neural network      |
| Python               | a cross-platform computer programming language               |
| Tensorflow           | a symbolic mathematics system based on dataflow programming is widely used in the programming of various machine learning algorithms |
| PyTorch              | an open source Python machine learning library, based on Torch |


### 1.4 References



### 1.5 Overview

The remaining sections of this document provide a general description of the project, including the main content of the study and the target users of our project,  the functional and nonfunctional requirements and the development environment of the project.

General description is discussed in section 2 of this document. Section 3 gives the functional requirements of our project. Features that users can use in our project can be clearly seen and they are described with text descriptions and activity diagrams. Section 4 gives the nonfunctional requirements. Section 5 is for our project development environment.

## 2 Overall Description

Our project is a research-oriented project, mainly studying the anomaly detection effect of various time series anomaly detection models on different data sets. Our main work is to read the relevant papers, try to reproduce and optimize the appropriate models, and finally present our research results in the form of the front-back end.

The target users of our project are university students and other researchers who are also engaged in time series anomaly detection research. We hope to find a model with excellent anomaly detection effect and optimize it through experimental research and comparison, so as to provide help for other related researchers in the future.

## 3 Functional Requirements

### 3.1 System Scope

The project provides a total of three models, including TSI-CNN and RNN models, and two datasets. Users can select different models to test different datasets, and can either select existing datasets or upload their own datasets. The system will call the pre-trained models on the backend for testing and the results will be displayed in the interface.

### 3.2 Systematic General Process

<img src="img/process.png" alt="process" style="zoom: 50%;" />

### 3.3 Requirements Analysis

#### Select Data Use Case

<img src="img/3.3.1.png" alt="3.3.1" style="zoom:67%;" />

#### Execute Algorithm Use Case

<img src="img/3.3.2.png" alt="3.3.2" style="zoom: 67%;" />

#### View Results Use Case

<img src="img/3.3.3.png" alt="3.3.3" style="zoom:67%;" />

#### Upload Dataset Use Case

<img src="img/3.3.4.png" alt="3.3.4" style="zoom:67%;" />

## 4 Nonfunctional Requirements

### 4.1 Performance Requirements

The performance requirements include the evaluate performance of the model and the performance of the front-end interaction. Since the core of the project is the research of anomaly detection model, therefore, we will mainly focuses on the former.

* The response time of the model evaluating should be limited to a certain time, but the specific response time limit has not been determined, which is connected to the length of each data segment in the dataset.
* The response time for the user to upload the dataset to the server should be less than 3s of each dataset.

### 4.2 Precision Requirements

The precision requirements mainly refer to the **prediction accuracy of the model for abnormal data** in each data set. It is necessary for us to read enough relevant papers and reproduce better models. We will only used the models whose accuracy are more than 85%.

### 4.3 Maintainability Requirements

Since our system will constantly add models and datasets, we need to make sure that we can add or modify models and datasets more easily without having to make many changes to the front end and back end code.

## 5 Project development environment

### 5.1 Device

A computer satisfies following software and hardware requirements:

- at least one CUDA-capable GPU
- NVIDIA Driver (version >= 410.48 for Linux x86_64 and version >= 411.31 for Windows x86_64)
- CUDA Toolkit 10.0 and above
- CPU Memory at least 1.0GB

### 5.2 Programming Language

**Python**

* for model erection and back-end code

* The main dependency packages include:
  * tensorflow
  * torch
  * pandas
  * numpy
  * argparse
  * matplotlib
  * pathlib
  * tqdm

**HTML, CSS, JavaScript**

* for front-end code

### 5.3 Supporting Software

**IDE**

* PyCharm 2019.3.5

**Collaborative Development Tools**

* **GitHub** for code version management and weekly work summary
* **Microsoft Todo** for work assignment
* **Postman** for the interface test
