# Comparative Analysis Modular, Sequential, and Parallel Neural Network Architectures on Inference Accuracy
**Author:** Aryan Kulkarni (Bellarmine College Preparatory)

I am a student at Bellarmine College Preparatory with a deep interest in the intersection of computational logic and machine learning. 
Currently, I am conducting independent research on neural network architecture efficiency, specifically comparing Sequential and Modular 
frameworks using PyTorch and TensorFlow. My goal is to understand how algorithmic optimization can reduce latency in complex data environments. 
I am always looking to learn from experts in the field to transition my findings into publishable academic research.

## Project Overview
This research evaluates the trade-offs between **Sequential**, **Parallel**, and **Modular (Neural Chains)** frameworks. 
Using PyTorch and TensorFlow, I measured how architectural complexity impacts:
1. **Efficiency:** epoch rates and Mean Squared Error (MSE).
2. **Latency:** Inference time across linear, curved, and chained datasets.

## Technical Tools Used
* **Frameworks:** PyTorch, Keras/TensorFlow
* **Optimization:** Adam Optimizer(Batch Normalization)
* **Environment:** Google Colab (A100/L4 GPUs)
* **Graphing:** Matplotlib

## Key Findings
* **Latency**: Through a graphed data analysis of Matplotlib (prediction/epoch rates), I found that the Modular (Neural Chain) architecture produced consistent results showing faster
* time rates than the Parallel and Sequential structures through all the linear, curved, and chain environments.
* **Prediction(MSE)**: While Modular structure was able to produce accruate results in standard scenarios, in the case of the most complex environment(Chained regression), the sequential model
* was able to produce a better predictive accuracy (lower MSE). This suggests that while in most scenarios the modular environment has better capabilities and environments,
* specifically within complex data scenarios, the sequential structure proves to be better than the Modular approach.













