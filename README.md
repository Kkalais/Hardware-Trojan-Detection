# Machine Learning Techniques for Hardware Trojan Detection 

### The problem

- Rapid development of technology drives companies to design and fabricate their ICs in non-trustworthy outsourcing foundries to reduce the cost
- There is space for a synchronous form of virus, known as Hardware Trojan (HT), to be developed. HTs leak encrypted information, degrade device performance or lead to total destruction.

### Description of CasLab-HT algorithm

- We used the design tool, Design Compiler NXT from Synopsys for the dataset's feature extraction
- The features consist via area and power characteristics of the circuits. In total they were used 50 area and power features.
- 7 Machine Learning models for the detection and classification of Trojan Free and Trojan Infected circuits, based on Gate Level Netlist phase and features for Application Specific Integrated Circuit (ASIC) circuits.

## Prerequisites
Install the libraries below used by the project by entering in console the following command:

  ```pip3 install pandas matplotlib keras scikit-learn numpy more-tertools seaborn xgboost```
  
Clone the repository locally by entering in console the following command:

  ```git clone https://github.com/Kkalais/Hardware-Trojan-Detection.git```
 
## Run
 
We are using **Gradient Boosting, XGBoost, Logistic Regression, K-Nearest Neighbors, Support-Vectors Machine, Random Forest, and Multilayer Perceptron Neural Network** to classify the samples into Trojan Free and Trojan Infected circuits.
 
In order to run the code using the above-mentioned algorithms just enter in console the following commands :
 
  ```python3 main.py gradient_goosting```
  
  ```python3 main.py xgboost```
 
  ```python3 main.py logistic_regression```
  
  ```python3 main.py k_neighbors```
  
  ```python3 main.py svm```
 
  ```python3 main.py random_forest```
 
  ```python3 main.py mlp```
  
respectively.

There is also a mode that runs all four algorithms consecutively, and produces a bar plot to compare the algorithms' results. Please enter in console:

```python3 main.py comparative```

## Authors

* **Konstantinos Kalais**, *Developer* 
