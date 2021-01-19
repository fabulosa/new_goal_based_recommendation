# Time Slice Inputation for Goal-based Recommendation

## Introduction:

This repo includes code for the paper:

* [Weijie Jiang](https://jennywjjiang.com/) and [Zachary A. Pardos](https://gse.berkeley.edu/zachary-pardos). 2019. [Time Slice Imputation for Personalized Goal-based Recommendation in Higher Education.](https://dl.acm.org/doi/10.1145/3298689.3347030) In Thirteenth ACM Conference on Recommender Systems (RecSys'19). ACM, pp 506-510,

which extended the RNN-based goal-based course recommendation algorithm in Section 7 of the following paper 

* [Jiang, W.](https://www.jennywjjiang.com), [Pardos, Z.A.](https://gse.berkeley.edu/zachary-pardos), Wei, Q. (2019) [Goal-based Course Recommendation.](https://dl.acm.org/doi/10.1145/3303772.3303814) In C. Brooks, R. Ferguson & U. Hoppe (Eds.) *Proceedings of the 9th International Conference on Learning Analytics and Knowledge* (LAK). ACM, Pages 36-45,

by enhancing the learning process the personalized prerequisite courses for any target course, and also applying the goal-based recommendation framework to the MOOCs context, as illustrated here:

![](illustration_figure.png)

## Dataset:

Due to FERPA privacy protection, we cannot publish the original student enrollment dataset and the MOOCs dataset. However, we provide code to run the proposed method on [a synthetic student enrollment dataset](https://github.com/fabulosa/goal-based-recommendation/tree/master/synthetic_data_samples) that we published in this [repo](https://github.com/fabulosa/goal-based-recommendation) with data descriptions.

## Steps for Running the Code:

### Environment Prerequisites:
* python3
* pytorch
* install other dependencies by: *pip3 install -r requirements.txt*

### Data Preprocessing:

**-- command:**

* Set up global parameters in _data\_preprocess/utils.py_
* `python data_preprocess/preprocess.py`
	
This command hard codes the locations of the expected data files to be in the synthetic data folder. This path can be changed in utils.py.

Then the following intermediate files will be generated for model training:
	
* **course dictionaries (_course\_id.pkl_)**: a pair of python dictionaries mapping courses to their preprocessed ID and vice versa.
* **grade dictionary (_grade\_id.pkl_)**: a pair of python dictionaries mapping all types of grades to their preprocessed ID and vice versa. 
* **semester dictionary (_semester\_id.pkl_)**: a pair of python dictionaries mapping semesters to their preprocessed ID vice versa. For example, the earliest semester in the dataset, 2014 Fall, will be set 0 as its ID. 
* **condensed student enrollments and grades (_stu\_sem\_major\_grade\_condense.pkl_)**: a 2D python list with dimention `n×m`, where `n` is the number of students and `m` is the number of semesters covered in the dataset: <img src="https://latex.codecogs.com/gif.latex?[s_1,&space;s_2,&space;s_3,&space;...,&space;s_n]" title="[s_1, s_2, s_3, ..., s_n]" />
, where <img src="https://latex.codecogs.com/gif.latex?s_i=[t_{i1},&space;t_{i2},&space;...,&space;t_{im}]" title="s_i=[t_{i1}, t_{i2}, ..., t_{im}]" />, and <img src="https://latex.codecogs.com/gif.latex?s_i" title="s_i" /> denotes the preprocessed enrollment histories of the i-th student in your data (multiple semesters) and <img src="https://latex.codecogs.com/gif.latex?t_{ik}" title="t_{ik}" /> represents the specific enrollment histories of the i-th student in the k-th semester. Note that the k-th semester of all the students refers to the same semester, for example, m=3, which means there are 3 semesters covered in your data: Fall 2019, Spring 2020, Summer 2020, then <img src="https://latex.codecogs.com/gif.latex?t_{i2}&space;(i=1,2,...,n)" title="t_{i2} (i=1,2,...,n)" /> will contain enrollment histories of Spring 2020 for all students in your data. <img src="https://latex.codecogs.com/gif.latex?t_{ik}=\{\}" title="t_{ik}=\{\}" /> (empty) if the i-th student did not enroll in any course in semester k.
The format of <img src="https://latex.codecogs.com/gif.latex?t_{ik}" title="t_{ik}" /> is a python dictionary: <img src="https://latex.codecogs.com/gif.latex?\{'major':&space;m_{ik},&space;'course\_grade':&space;[(c_{ik}^1,&space;g_{ik}^1),(c_{ik}^2,&space;g_{ik}^2),...,(c_{ik}^p,&space;g_{ik}^p)]\}" title="\{'major': m_{ik}, 'course\_grade': [(c_{ik}^1, g_{ik}^1),(c_{ik}^2, g_{ik}^2),...,(c_{ik}^p, g_{ik}^p)]\}" />, where <img src="https://latex.codecogs.com/gif.latex?m_{ik}" title="m_{ik}" /> refers to the major ID of the i-th student's major in the k-th semester, and <img src="https://latex.codecogs.com/gif.latex?(c_{ik}^p,&space;g_{ik}^p)" title="(c_{ik}^p, g_{ik}^p)" /> refers to the course ID of the p-th course the i-th student enrolled and the grade ID received for that course in the k-th semester. 




