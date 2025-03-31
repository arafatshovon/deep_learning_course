
# Atrial Fibrillation Detection via Contactless Radio Monitoring and Knowledge Transfer

## 👉Folder Structure Overview

This folder contains the following items:
- **data**

  - This folder contain experiment data including synchronised reference ECG signals and "Cardiac mechanical motion signals" extracted from raw radio signals using the signal processing algorithms decscribed in the paper. 
  - The data provided here is a partial disclosure, containing recordings from 20 subjects. Each recording is divided into 3 distinct samples, with a 10-second interval, resulting in a total of 60 samples. For example, in the file 'A_0001_1.mat', 'A_0001' represents the unique subject ID, and '1' indicates the 1th divided sample. 
  - The label information is summarized in 'labels.csv', where samples with AF are denoted by '1', and those without are denoted by '0'.
  
- **Teacher_Student_neural_network**: 
  - `dataset.py`
  - `Student_net.py`: Student network.
  - `Teacher_net.py`: Teacher network.
  - `test.py`: test code.

- `README.md`
- `requirements.txt`


## 👨‍💻 Installation
Install Python 3.9 and necessary dependencies.
```shell
pip install -r requirements.txt 
```


## 🚀Teacher-Student neural network
### 📜 Data
The data required for testing is in the 'Data' folder in the current directory, where the 'Radar_data' in each '.mat' file is the input to our network.
### 📜Test
To run the test code, just run the following command:
```shell
cd Teacher_Student_neural_network
```
```shell
python test.py 
```
### 📜Result
After the code is run, the network will output the predicted value of each sample and determine whether the sample is atrial fibrillation. Finally, the code will analyze and summarize the diagnostic results for all samples in the './Data' directory (corresponding to the sequence-level diagnostic strategy in the paper) and the diagnostic results for all subjects (corresponding to the set-level diagnostic strategy in the paper).



## ❤️ Acknowledgments
- [Contactless Electrocardiogram Monitoring
With Millimeter Wave Radar](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9919401)
- [Practical intelligent diagnostic algorithm for wearable 12-lead ECG via self-supervised learning on large-scale dataset](https://www.nature.com/articles/s41467-023-39472-8)
- [THE MODALITY FOCUSING HYPOTHESIS](https://openreview.net/pdf?id=w0QXrZ3N-s)
- [Interpretable deep learning for automatic
diagnosis of 12-lead electrocardiogram](https://www.cell.com/iscience/fulltext/S2589-0042(21)00341-2?ref=https://githubhelp.com)


