# DEEP LEARNING FOR ANOMALY DETECTION: A SURVEY

[toc]

### 1. Basic Definition of Anomaly Detection

##### 1.1 Anomaly and Anomaly Detection

In the beginning, the paper talks about the definition of anomaly and anomaly detection. According to the paper, anomalies are also referred to as abnormalities, deviants, or outliers in the data mining and statistics literature. And based on the relationship between the anomaly points and other points, anomalies are classified into several categories and the novelties are defined so that outliers can be distinguished.

Additionally, some basic knowledge of anomaly detection is mentioned in this paper. According to the paper, the goal of anomaly detection is to determine all such instances in a data-driven fashion. Given that the workload of anomaly detection is quite large, machine learning methods are often needed to assist the process of anomaly detection.

##### 1.2 Deep Learning and Deep Anomaly Detection

And in the recently years, deep learning, a kind of machine learning with better efficiency and flexibility has been increasingly widely applied to anomaly detection. Anomaly detection with deep learning method is called Deep Anomaly Detection (DAD). Nowadays, the volume of data is increasing and the boundary between normal and anomalous (erroneous) behaviour is often not precisely defined in several data domains, so traditional methods with bad efficiency, scalability and adaptation cannot meet the requirement of anomaly detection. So deep anomaly detection is born at the right moment and related theories are rapidly developing.

### 2. Different aspects of deep learning-based anomaly detection

##### 2.1 Nature of Input Data

The choice of a deep neural network architecture in deep anomaly detection methods primarily depends on the nature of input data. Input data can be broadly classified into sequential or non-sequential data. Sequential data is a sequence of values and the order among the values is of vital importance and non-sequential data is what does not belong to sequential data.

##### 2.2 Availability of Labels

Labels indicate whether a chosen data instance is normal or an outlier. Deep anomaly detection models can be broadly classified into three categories based on the extent of availability of labels: supervised deep anomaly detection, semi-supervised deep anomaly detection and unsupervised deep anomaly detection.

##### 2.3 Training Objective

Based on training objectives employed there are two new categories of deep anomaly detection techniques: deep hybrid models (DHM) and one class neural networks (OC-NN).

##### 2.4 Type of Anomaly

Anomalies can be broadly classified into three types: point anomalies, contextual anomalies and collective anomalies. Point anomalies often represent an irregularity or deviation that happens randomly and may have no particular interpretation. A contextual anomaly is also known as the conditional anomaly is a data instance that could be considered as anomalous in some specific context. Contextual anomaly is identified by considering both contextual and behavioural features. Anomalous collections of individual data points are known as collective or group anomalies, wherein each of the individual points in isolation appears as normal data instances while observed in a group exhibit unusual characteristics. Deep anomaly detection methods have been shown to detect all three types of anomalies with great success. 

##### 2.5 Output of DAD Techniques

A critical aspect for anomaly detection methods is the way in which the anomalies are detected. Generally, the outputs produced by anomaly detection methods are either anomaly score or binary labels. Anomaly score describes the level of outlier‘s intensity for each data point. The data instances may be ranked according to anomalous score, and a domain-specific threshold will be selected by subject matter expert to identify the anomalies. Instead of assigning scores, some techniques may assign a category label as normal or anomalous to each data instance. In general, decision scores reveal more information than binary labels.

### 3. Applications of Deep Anomaly Detection

##### 3.1 Intrusion Detection

The intrusion detection system (IDS) refers to identifying malicious activity in a computer-related system. IDS may be deployed at single computers known as Host Intrusion Detection (HIDS) to large networks and Network Intrusion Detection (NIDS). HIDS systems are installed software programs which monitors a single host or computer for malicious activity or policy violations by listening to system calls or events occurring within that host. NIDS systems deal with monitoring the entire network for suspicious traffic by examining each and every network packet.

##### 3.2 Fraud Detection

Fraud is a deliberate act of deception to access valuable resources. Fraud detection refers to the detection of unlawful activities across various industries. Detecting and preventing fraud is not a simple task since fraud is an adaptive crime. Many traditional machine learning algorithms have been applied successfully in fraud detection. The challenge associated with detecting fraud is that it requires real-time detection and prevention. Deep anomaly detection techniques mainly solve these kinds of fraud, including banking fraud, mobile cellular network fraud, insurance fraud and healthcare fraud.

##### 3.3 Malware Detection

Malware refers to malicious software. In order to protect legitimate users from malware, machine learning based efficient malware detection methods are proposed. The challenge associated in malware detection problems is the sheer scale of data. Furthermore, the malware is very adaptive in nature, wherein the attackers would use advanced techniques to hide the malicious behaviour. And some DAD techniques proposed recently can address these challenges effectively and detect malware.

##### 3.4 Medical Anomaly Detection

Several studies have been conducted to understand the theoretical and practical applications of deep learning in medical and bio-informatics. Deep learning based architectures are employed with great success to detect medical anomalies. The vast amount of imbalanced data in medical domain presents significant challenges to detect outliers. Even though deep learning models produce outstanding performance, these models lack interpret-ability. Fortunately, In recent times there are models with good interpret-ability proposed.

##### 3.5 Deep learning for Anomaly detection in Social Networks

Anomalies in a social network are irregular often unlawful behaviour pattern of individuals within a social network. Detecting these irregular patterns is of prime importance since if not detected, the act of such individuals can have a serious social impact. The heterogeneous and dynamic nature of data presents significant challenges to DAD techniques. Despite these challenges, several DAD techniques proposed recently are shown outperform state-of-the-art methods.

##### 3.6 Log Anomaly Detection

Anomaly detection in log file aims to find text, which can indicate the reasons and the nature of the failure of a system. The limitation of such approaches is that newer messages of failures are easily are not detected. The unstructured and diversity in both format and semantics of log data pose significant challenges to log anomaly detection. Anomaly detection techniques should adapt to the concurrent set of log data generated and detect outliers in real time.

##### 3.7 Internet of things (IoT) Big Data Anomaly Detection

Anomaly detection in the IoT networks identifies fraudulent, faulty behaviour of these massive scales of interconnected devices. The challenges associated with outlier detection is that heterogeneous devices are interconnected which renders the system more complex.

##### 3.8 Industrial Anomalies Detection

Damage to industrial systems not only causes economic loss but also a loss of reputation, therefore detecting and repairing them early is of utmost importance. Damages caused to equipment are rare events, thus detecting such events can be formulated as an outlier detection problem. The challenges associated with outlier detection in this domain is both volumes as well as the dynamic nature of data since failure is caused due to a variety of factors.

##### 3.9 Anomaly Detection in Time Series

Data recorded continuously over duration is known as time series. Time series data can be broadly classified into univariate and multivariate time series. In case of univariate time series, only single variable (or feature) varies over time. Types of anomalies in univariate and multivariate time series are categorized into following groups: Point Anomalies, Contextual Anomalies and Collective Anomalies. Also there are some challenges in time series based anomaly detection. For example, noise within the input data seriously affects the performance of algorithms. Further more, time series data is usually dynamically evolving, which makes detecting anomalies in real time is necessary.

 In recent times, many deep learning models have been proposed for detecting anomalies within univariate and multivariate time series data. The advancements in deep learning domain offer opportunities to extract rich hierarchical features which can greatly improve outlier detection within univariate time series data. And despite that anomaly detection within multivariate time series data is a challenging task, effective multivariate anomaly detection enables fault isolation diagnostics.

##### 3.10 Video Surveillance

Video Surveillance also popularly known as Closed-circuit television (CCTV) involves monitoring designated areas of interest in order to ensure security. Video surveillance applications have been modelled as anomaly detection problems owing to lack of availability of labelled data. The lack of an explicit definition of an anomaly in real-life video surveillance is a significant issue that hampers the performance of DAD methods as well.

### 4. Deep Anomaly Detection Models

##### 4.1 Supervised Deep Anomaly Detection

Deep supervised learning methods depend on separating data classes whereas unsupervised techniques focus on explaining and understanding the characteristics of data. In general, supervised deep learning based classification schemes for anomaly detection have two sub-networks, a feature extraction network followed by a classifier network. The computational complexity of deep supervised anomaly detection methods based techniques depends on the input data dimension and the number of hidden layers trained using back-propagation algorithm.

Supervised anomaly detection techniques are superior in performance compared to unsupervised anomaly detection techniques. However, multi-class supervised techniques require accurate labels for various normal classes and anomalous instances, which is often not available.

##### 4.2 Semi-supervised Deep Anomaly Detection

Semi-supervised techniques assume that all training instances have only one class label. Semi-supervised DAD methods proposed to rely on proximity and continuity to score a data instance as an anomaly. Similar to supervised DAD techniques, The computational complexity of semi-supervised DAD methods based techniques primarily depends on the dimensionality of the input data and the number of hidden layers used for representative feature learning.

According to the paper, Generative Adversarial Networks (GANs) trained in semi-supervised learning mode have shown great promise, even with very few labelled data. However, the hierarchical features extracted within hidden layers may not be representative of fewer anomalous instances hence are prone to the over-fitting problem.

##### 4.3 Hybrid Deep Anomaly Detection

In deep hybrid models, the representative features learned within deep models are input to some specific traditional algorithms. Building a robust anomaly detection model on complex, high-dimensional spaces require feature extractor and an anomaly detector. The computational complexity of a hybrid model includes the complexity of both deep architectures as well as traditional algorithms used within.

The feature extractor significantly reduces the ‘curse of dimensionality’, which makes hybrid models more scalable and computationally efficient. However, the hybrid approach is suboptimal because it is unable to influence representational learning within the hidden layers of feature extractor since generic loss functions are employed instead of the customized objective for anomaly detection.

##### 4.4 One-class Neural Networks (OC-NN) for Anomaly Detection

One-class neural networks (OC-NN) combines the ability of deep networks to extract a progressively rich representation of data along with the one-class objective to separate all the normal data points from the outliers. Performs combined representation learning and produces an outlier score for a test data instance. The computational complexity of an OC-NN model includes only the complexity of the deep network of choice and OC-NN models do not require data to be stored for prediction, which makes memory complexity very low. However, training time of OC-NN is proportional to the input dimension.

OC-NN propose an alternating minimization algorithm for learning the parameters of the OC-NN model and the subproblem of the OC-NN objective is equivalent to a solving a quantile selection problem which is well defined. However, training times and model update time are proportional to the input dimension, so they may be longer for high dimensional input data and given the change in input space model updates would also take longer time. 

##### 4.5 Unsupervised Deep Anomaly Detection

Unsupervised deep anomaly detection does not need labelled data, but it depends on that the “normal” regions in the original or latent feature space can be distinguished from ”anomalous” regions in the original or latent feature space and the majority of the data instances are normal compared to the remainder of the data set. Unsupervised anomaly detection algorithm produces an outlier score of the data instances based on intrinsic properties of the data-set such as distances or densities, which will be captures by the hidden layers of deep neural network. The computational complexity of model depends on three parts: the number of operations, network parameters and hidden layers.

Unsupervised Deep Anomaly Detection is a cost effective technique to find the anomalies since it does not require annotated data for training the algorithms. And the technique which Learns the inherent data characteristics to separate normal from an anomalous data point identifies commonalities within the data and facilitates outlier detection. However, the computational complexity of training an autoencoder is much higher than some traditional methods, so often it is challenging to learn commonalities within data in a complex and high dimensional space. And the unsupervised technique are very sensitive to noise and data corruptions, so they are often less accurate than supervised or semi-supervised techniques.

##### 4.6 Miscellaneous Techniques

###### 4.6.1 Transfer Learning based anomaly detection

Transfer learning is an essential tool in machine learning to solve the fundamental problem of insufficient training data. It aims to transfer the knowledge from the source domain to the target domain by relaxing the assumption that training and future data must be in the same feature space and have the same distribution.

###### 4.6.2 Zero Shot learning based anomaly detection

Zero shot learning (ZSL) aims to recognize objects never seen before within training set. This setting is important in the real world since one may not be able to obtain images of all the possible classes at training.

###### 4.6.3 Ensemble based anomaly detection

Autoencoder ensembles consisting of various randomly connected autoencoders are experimented to achieve promising results on several benchmark datasets. The ensemble approaches are still an active area of research which has been shown to produce improved diversity, thus avoid overfitting problem while reducing training time.

###### 4.6.4 Clustering based anomaly detection

Clustering involves grouping together similar patterns based on features extracted detect new anomalies. Deep learning enabled clustering approach anomaly detection utilizes models to get the semantical presentations of normal data and anomalies to form clusters and detect outliers. 

###### 4.6.5 Deep Reinforcement Learning (DRL) based anomaly detection

Deep reinforcement learning (DRL) methods have attracted significant interest due to its ability to learn complex behaviours in high-dimensional data space. The DRL based anomaly detector does not consider any assumption about the concept of the anomaly, the detector identifies new anomalies by consistently enhancing its knowledge through reward signals accumulated.

###### 4.6.6 Statistical techniques deep anomaly detection

Some statistical techniques can play a positive role in processing real-time data. The algorithms combining the ability of wavelet analysis, neural networks and these kinds of statistical techniques in a sequential manner to detect real-time anomalies are shown to be a very promising.

### 5. Deep neural network architectures for locating anomalies

##### 5.1 Deep Neural Networks (DNN)

Deep architectures overcome the limitations of traditional machine learning approaches of scalability, and generalization to new variations within data and the need for manual feature engineering. Deep Belief Networks (DBNs) are class of deep neural network which comprises multiple layers of graphical models known as Restricted Boltzmann Machine (RBMs). DBNs are shown to scale efficiently to big-data and improve interpretability.

##### 5.2 Spatiotemporal Networks (STN)

Spatiotemporal Networks (STNs) comprises of deep neural architectures combining both CNN’s and LSTMs to extract spatiotemporal features. The temporal features (modelling correlations between near time points via LSTM), spatial features (modelling local spatial correlation via local CNN’s) are shown to be effective in detecting outliers.

##### 5.3 Sum-Product Networks (SPN)

Sum-Product Networks (SPNs) are directed acyclic graphs with variables as leaves, and the internal nodes, and weighted edges constitute the sums and products. SPNs are considered as a combination of mixture models which have fast exact probabilistic inference over many layers. SPNs are more traceable over high treewidth models without requiring approximate inference. Furthermore, SPNs are shown to capture uncertainty over their inputs yielding robust anomaly detection.

##### 5.4 Word2vec Models

Word2vec is a group of deep neural network models used to produce word embeddings. These models are capable of capturing sequential relationships within data instance. Obtaining word embedding features as inputs are shown to improve the performance in several deep learning architectures.

##### 5.5 Generative Models

Generative models aim to learn exact data distribution in order to generate new data points with some variations. The two most common and efficient generative approaches are Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN). Some generative models proposed are shown to be effective in identifying anomalies on high dimensional and complex datasets.

##### 5.6 Convolutional Neural Networks (CNN)

Convolutional Neural Networks are the popular choice of neural networks for analysing visual imagery. CNN’s ability to extract complex hidden features from high dimensional data with complex structure has enabled its use as feature extractors in outlier detection for both sequential and image dataset.

##### 5.7 Sequence Models

Recurrent Neural Networks (RNNs) are shown to capture features of time sequence data. Long Short-Term Memory (LSTM) networks, a particular type of RNNs comprising of a memory cell that can store information about previous time steps were introduced to capture the context as time steps increases. And Long Short- Term Memory neural network based algorithms for anomaly detection have been reported to produce significant performance gains over conventional methods.

##### 5.8 Autoencoders

Autoencoders are artificial neural networks used in semi-supervised and unsupervised learning. Autoencoders with single layer along with a linear activation function are nearly equivalent to Principal Component Analysis (PCA) and autoencoders enable both linear or nonlinear transformations. Autoencoders represent data within multiple hidden layers by reconstructing the input data, effectively learning an identity function. However, the autoencoders trained solely on normal data instances fail to reconstruct the anomalous data samples, therefore, producing a large reconstruction error. And due to noisy training data the performance of autoencoders might get degraded. But anyway, autoencoders are simple and effective architectures for outlier detection which have been fairly popular in recent years.