# Deep Learning Framework for Vessel Trajectory Prediction Using Auxiliary Tasks and Convolutional Networks

This is a PyTorch implementation of AIS-ACNet in the paper entitled "Deep Learning Framework for Vessel Trajectory Prediction Using Auxiliary Tasks and Convolutional Networks"
The paper is accepted at EAAI (Engineering Applications of Artificial Intelligence) in Jan. 2024.

## Abstract 
With the exponential growth in vessel traffic and the increasing complexity of maritime operations, there is a pressing need for reliable and efficient methods to forecast vessel movements. The accurate prediction of vessel trajectories plays a pivotal role in various maritime applications, including route planning, collision avoidance, and maritime traffic management. Traditional statistical and machine learning approaches have shown limitations in capturing the complex spatial-temporal patterns of vessel movements. Deep learning techniques have emerged as a promising solution due to their ability to handle large-scale datasets and capture nonlinear relationships. This study proposes a novel deep learning-based vessel trajectory prediction framework for AIS data using Auxiliary tasks and Convolutional encoders (AIS-ACNet). The framework utilizes various features of Automatic Identification System (AIS) data, including geographical positions, and vessel dynamics such as Speed Over Ground (SOG), and Course Over Ground (COG), for trajectory prediction. The AIS-ACNet employs parallel convolutional encoder networks with feature fusion layers to control the weight of auxiliary features. The model is trained with a multi-task learning objective that includes auxiliary SOG and COG prediction tasks. This framework enhances the model's vessel trajectory prediction performance by efficiently incorporating vessel dynamics. The proposed framework is evaluated on a real-world AIS dataset retrieved from the Port of Houston, Texas, USA. The result shows that AIS-ACNet achieves 5.31% increase in average displacement error compared to the best performing baseline model. Also, the model demonstrates the ability to perform robustly on various types of trajectories. 


## Model Architecture

![architecture_r2](https://github.com/yuyolshin/AIS-ACNet/assets/31876093/e7b6658a-36ba-4643-9c20-09af15201a38)

## Performance Comparison 
#### Dataset

- AIS data from Port of Houston, Texas, USA (2017.01)

![AIS_figure_houston](https://github.com/yuyolshin/AIS-ACNet/assets/31876093/d159dc4c-428c-41d3-ba68-e2894c402b4a)

AIS trajectories in the coastal area of Port of Houston on January 4th, 2017

#### Results
![image](https://github.com/yuyolshin/AIS-ACNet/assets/31876093/4d7097bc-0c52-4e7c-a939-7da14f201143)

Evaluation results on real-world AIS datasets show that our model consistently outputs of baseline models .

##
##### code implementation
Code for AIS-ACNet has been implemented by modifying codes from Graph WaveNet (https://github.com/nnzhan/Graph-WaveNet) [1] and Sekhon and Flemming (2020) (https://github.com/coordinated-systems-lab/VesselIntentModeling) [2]

[1] Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph wavenet for deep spatial-temporal graph modeling. arXiv preprint arXiv:1906.00121.

[2] Sekhon, J., & Fleming, C. (2020, July). A spatially and temporally attentive joint trajectory prediction framework for modeling vessel intent. In Learning for Dynamics and Control (pp. 318-327). PMLR.
