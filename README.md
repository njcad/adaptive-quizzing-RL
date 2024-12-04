### Can Q-Learning Learn You-Learning? Reinforcement Learning for Adaptive Quizzing ###
A project for Stanford CS238.
December 2024.

### Data ###
The data for this project is sourced from kaggle, at https://www.kaggle.com/competitions/riiid-test-answer-prediction/data. 
In order to run the src/preprocess.py, you need to download that dataset and set the path thereto. 
This will create a sar_space.csv on which the src/qlearning.py will learn. 

### Performance ###
The Q-Learning algorithm produces a policy (policies/v4.policy) which has a 14% increase over random baseline in expected reward.
