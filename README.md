# Perceptron
A simple perceptron in Python used to predict a robot's movements.  

**Activation function:** Sigmoid  
**Optimisation:** Gradient Descent  

### Data
Time-series data of a robot's position is provided. This is split into 3 columns as shown below.  
The target labels are given by the robot's next position in the data-set.

|  col1   |  col2   |  col3  |
| ------- | ------- | ------ |
|    0    |    0    |0.100502|
|    0    |0.100502 |0.161233|
|0.100502 |0.161233 |0.215614|
|0.161233 |0.215614 |0.2647  |
|0.215614 |0.2647   |0.308606|
|0.2647   |0.308606 |0.347805|
|0.308606 |0.347805 |0.382622|
|0.347805 |0.382622 |0.413436|
|0.382622 |0.413436 |0.440748|
|0.413436 |0.440748 |0.464818|


### Requirements
- [Numpy](https://github.com/numpy/numpy)  
- [Pandas](https://github.com/pandas-dev/pandas)  
- [Matplotlib](https://github.com/matplotlib/matplotlib)  
