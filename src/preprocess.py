"""
Original data structure is not well-suited to a state action reward space.
This is a file to preprocess the data into a reasonable SAR space.

States: tuples of [part, num_correct_so_far, num_incorrect_so_far]. Part represents
the part of the test these questions come from, representing the approximate domain.

Actions: ask an easy, medium, or hard question.

Rewards: rewards should fit the difficulty of a question. Getting a question wrong
should have a small positive reward, since it still definitely contributes to learning.
Harder questions should have a greater reward, right or wrong. We can model question 
difficulty by simply observing the fraction of times it is answered correctly. Then, 
rewards are proportional to difficulty.
"""

import pandas as pd 
import os
import json
import csv
import math 
from tqdm import tqdm


def eval_question_difficulty(data):
    """
    Track how often questions are answered correctly.
    """
    # question ID : (num times correct, num times asked)
    question_difficulty = {}

    for _, row in tqdm(data.iterrows()):
        # assert that this user event is a question
        if row['content_type_id'] == 0:
            qID = row['content_id']

            # initialize to zero tuple
            if qID not in question_difficulty.keys():
                question_difficulty[qID] = [0, 0]

            # update based on correctness
            question_difficulty[qID][1] += 1
            if row['answered_correctly']:
                question_difficulty[qID][0] += 1

    # save to json
    with open("../question_data_jsons/question_difficulty.json", 'w') as f:
        json.dump(question_difficulty, f)


def sort_questions(question_difficulty):
    """
    From empirical question responses, sort into levels 1 (easiest) to 5 (hardest).
    """
    # to avoid 0s in rare questions
    eps = 0.01

    # calculate the scores as approx num times correct / num times asked
    question_scores = []
    for qID in question_difficulty.keys():
        score = (question_difficulty[qID][0] + eps) / (question_difficulty[qID][1] + eps)
        question_scores.append((qID, score))
    
    # sort into fifths
    question_scores.sort(key=lambda x: x[1], reverse=True)
    num_questions = len(question_scores)
    fifth = num_questions // 5

    # maintain dict of qID : difficulty level (1 - 5)
    question_types = {}
    def build_dict(difficulty_list, type):
        for i in range(len(difficulty_list)):
            qID = difficulty_list[i][0]
            question_types[qID] = type 

    for i in range(5):
        questions = question_scores[i * fifth: (i + 1) * fifth if i < 4 else num_questions]
        build_dict(questions, i + 1)

    # save to json
    with open("../question_data_jsons/qID_to_level.json", 'w') as f:
        json.dump(question_types, f)


def build_qID_to_part(question_metadata_df):
    """
    Convert question metadata to dict of qID : part of test
    """
    qID_to_part = {}
    for _, row in question_metadata_df.iterrows():
        qID_to_part[row['question_id']] = row['part']

    with open("../question_data_jsons/qID_to_part.json", 'w') as f:
        json.dump(qID_to_part, f)


def sigmoid(x):
    """
    Helper function to calculate sigmoid to normalize reward difficulty score to (0, 1)
    """
    return 1 / (1 + math.exp(-x))


def get_reward(correct: bool, question_data, num_correct_so_far, num_incorrect_so_far):
    """
    Reward is proportional to question difficulty.
    Correct on easy is less than correct on hard.
    Incorrect on hard still less than correct on easy.
    Factor in student progress so far: if student is struggling, scale reward up, 
    reflecting that the quiz is too hard so far. If student is doing well, scale reward 
    down, reflecting that the quiz is too easy so far. 
    """
    # calculate empirical difficulty score for given question in range (0, 1]
    eps = 0.01
    score = (question_data[0] + eps) / (question_data[1] + eps)

    # make difficulties pronounced and flip with -log, and normalize with sigmoid 
    # harder questions have higher difficulty score 
    norm_difficulty = sigmoid(-math.log(score))

    # calculate student progress to scale reward up/down if student is doing too poorly/well
    progress_factor = 1 - (num_correct_so_far / (num_correct_so_far + num_incorrect_so_far + 1))

    # calculate reward from correctness, difficulty and student progress
    base_correct = 100
    base_incorrect = 1
    if correct:
        reward = base_correct * norm_difficulty * progress_factor
    else:
        reward = base_incorrect * norm_difficulty * progress_factor

    # round to 3 decimal places for space efficiency
    return round(reward, 3)


def build_SAR_space(data, qID_to_part, qID_to_level, question_difficulty):
    """
    Convert our data to a state, action, reward, state-prime space.
    state: [quiz_part, num_correct_so_far, num_incorrect_so_far]
    """
    output_file = "../sar_spaces/sar_space.csv"
    header = ['s', 'a', 'r', 'sp']    

    with open(output_file, mode='w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # track correct and incorrect and current part so far for each user
        num_correct_so_far = 0
        num_incorrect_so_far = 0
        current_part = 0
        current_student = -1

        for _, row in tqdm(data.iterrows()):
            # reset correct counts and part if new user
            if row['user_id'] != current_student:
                num_correct_so_far, num_incorrect_so_far, current_part = 0, 0, 0
                current_student = row['user_id']

            # only consider quiz question events
            if row['content_type_id'] == 0:
                qID = str(row['content_id'])

                # get the relevant part of the test; reset if new part
                part = qID_to_part[qID]
                if part != current_part:
                    num_correct_so_far, num_incorrect_so_far = 0, 0
                    current_part = part

                # set the current state
                state = [part, num_correct_so_far, num_incorrect_so_far]

                # set the current action (1 - 5) from difficulty level
                level = qID_to_level[qID]
                action = level 

                # update correct counts for next state
                correct = bool(row['answered_correctly'])
                if correct:
                    num_correct_so_far += 1
                else:
                    num_incorrect_so_far += 1
                next_state = [part, num_correct_so_far, num_incorrect_so_far]

                # get reward
                reward = get_reward(correct, question_difficulty[qID], num_correct_so_far, num_incorrect_so_far)

                # write to csv
                sarsp = [state, action, reward, next_state]
                writer.writerow(sarsp)


def main():
    """
    Main driver function to preprocess data and build state, action, reward space.
    """
    # declare paths
    data_path = "../data/train.csv"
    q_dif_path = "../question_data_jsons/question_difficulty.json"
    qID_to_level_path = "../question_data_jsons/qID_to_level.json"
    qID_to_part_path = "../question_data_jsons/qID_to_part.json"
    SAR_space_path = "../sar_spaces/sar_space.csv"
    question_metadata_path = "../data/questions.csv"

    # load in original dataset
    data = pd.read_csv(data_path)
    columns_needed = ['content_type_id', 'content_id', 'answered_correctly', 'user_id']
    data = data[columns_needed]
    # TESTING
    # data = data.head(100)
    print("Data loaded...")

    # eval difficulty
    if not os.path.exists(q_dif_path):
        eval_question_difficulty(data)
    print("Question difficulty evaluated...")

    # sort the questions by levels 1 (easiest) to 5 (hardest)
    if not os.path.exists(qID_to_level_path):
        with open(q_dif_path, 'r') as f:
            question_difficulty = json.load(f)
        sort_questions(question_difficulty)
    print("Question difficulty levels assigned...")

    # get question to part from metadata
    if not os.path.exists(qID_to_part_path):
        question_metadata_df = pd.read_csv(question_metadata_path)
        question_metadata_df = question_metadata_df['question_id', 'part']
        build_qID_to_part(question_metadata_df)
    print("Question quiz parts assigned...")

    # build state, action, reward space
    if not os.path.exists(SAR_space_path):
        with open(qID_to_level_path, 'r') as f:
            qID_to_level = json.load(f)
        with open(qID_to_part_path, 'r') as f:
            qID_to_part = json.load(f)        
        with open(q_dif_path, 'r') as f:
            question_difficulty = json.load(f)
        print("BUILDING...")
        build_SAR_space(data, qID_to_part, qID_to_level, question_difficulty)
    print("State, action, reward space built!")
    

if __name__ == "__main__":
    main()
