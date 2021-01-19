__author__ = 'jwj'
import torch
from torch.autograd import Variable
import myLSTM1
import pickle
import numpy as np
import sys
import csv
sys.path.insert(0, '/research/jenny/RNN3/train')
torch.manual_seed(1)    # reproducible


# data: student histories.
def process_data(data, dim_input_course, dim_input_grade):
    length = len([2 for i in data if i != []])
    input_pad = np.zeros((1, length, dim_input_course + dim_input_grade), int)  # padded input
    # label_padded = np.zeros((1, int(max_seq)-1, dim_input_grade), int)  # padded label
    sem_flag = 0
    stu_course_grade = []
    courses_histories = []
    for k in data:
        if k != []:
            stu_course_grade.append('sem ' + str(sem_flag) + ': ')
            for s in k:
                if s[1] != 5:  # not F grade, consider as enrolled successfully
                    courses_histories.append(s[0])
                stu_course_grade.append(id_course[s[0]] + '-' + id_grade[s[1]] + ' ')
                if s[1] <= 2:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4] = 1
                elif s[1] <= 5:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4 + 1] = 1
                elif s[1] == 6:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4 + 2] = 1
                elif s[1] == 7:
                    input_pad[0, sem_flag, dim_input_course + s[0] * 4 + 3] = 1
                if sem_flag != 0:  # the first semesters courses won't be in the input.
                    input_pad[0, sem_flag - 1, s[0]] = 1
                #elif s[1] != 5:  # add the first semesters courses to enrollment histories
                    #courses_sem1.append(s[0])
            sem_flag += 1
    # write the student's grade histories
    with open(target_course+"-top"+str(N), 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(stu_course_grade)

    # return the input for RNN, length, and the courses the student has not already taken
    #courses = np.where(np.sum(input_pad[0, :, :dim_input_course], axis=0) == 0)[0]
    courses_not_taken = set(id_course.keys()) - set(courses_histories)
    #print(courses_sem1)
    return input_pad, [length], courses_not_taken


def evaluate(label, recommended_course):
    if set(label).intersection(set(recommended_course)) != set():
        return 1
    else:
        return 0


# select students with enrollment records in Fall 2016 (target course period), no Summer 2016, Spring 2016 (vali1), and at least one semester's histories
def select_evaluation_set(course_id):
    f = open('/research/jenny/RNN/data_preprocess/stu_sem_grade_condense.pkl', 'rb')
    data = pickle.load(f)['stu_sem_grade_condense']
    data = np.array(data)
    vali_period = data[:, 24]
    stu_0 = []  # >b or pass
    stu_1 = []  # <b or no pass
    for i in range(len(data)):
        if vali_period[i] != [] and data[i, 23] == [] and data[i, 22] != []:  # had courses in Fall 2016, didn't have courses in Summer 2016
            num = len([1 for j in data[i, :22] if j != []])
            if num >= 1:  # have no fewer than one semester histories
                stu_sem_course = np.array(vali_period[i])[:, 0]  # get all the courses he selected in Fall 2016
                if course_id in stu_sem_course:  # if target course is in
                    where = np.where(stu_sem_course == course_id)[0]  # get the grade
                    grade = np.array(vali_period[i])[where, 1]
                    if grade in [1, 2]:
                        stu_0.append(i)

                    elif grade in [3, 4, 5]:
                        stu_1.append(i)

    print(len(data[stu_0]), len(data[stu_1]))
    #exit()
    if len(data[stu_0]) > 0 and len(data[stu_1]) > 0:
        return data[stu_0, :25], data[stu_1, :25]
    elif len(data[stu_0]) == 0:
        return [], data[stu_1, :25]
    elif len(data[stu_1]) == 0:
        return data[stu_0, :25], []


 # pick up top N according to plan_courses_1 and plan_courses_2
def top_N(plan_courses_value):
    course_grade = plan_courses_value.squeeze(0).squeeze(0)[dim_input_course:].view(-1, 4)[:, 0]  # only compare the value on letter grade>threshold
    sort, sorted_id = torch.sort(course_grade, descending=True)
    return sorted_id[:N]


def train(student):
    count_1 = 0
    for i in range(len(student)):
        print("Training Begin: the " + str(i) + 'th student')
        with open(target_course+"-top"+str(N), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['student' + str(i)])
        # process student data
        processed_data = process_data(student[i, :22], dim_input_course, dim_input_grade)
        padded_input = Variable(torch.Tensor(processed_data[0]), requires_grad=False).cuda()
        seq_len = processed_data[1]
        courses_not_taken = processed_data[2]

        # each student a model
        lstm = myLSTM1.long_term_LSTM(model1, model2, dim_input_course, dim_input_grade, target_course)
        optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
        epoch = 0
        old_loss = None
        while True:

            epoch += 1
            # clear gradients
            optimizer.zero_grad()

            y_pred = lstm(padded_input, seq_len, courses_not_taken, 23).cuda()
            loss = lstm.loss(y_pred).cuda()
            # recommended courses for 2 semesters
            plan_courses_1 = top_N(lstm.plan_courses_1)
            plan_courses_1_names = [id_course[int(i.cpu())] for i in plan_courses_1]

            real_courses_1 = np.array(student[i, 22])[:, 0]
            real_courses_1_names = [id_course[i] for i in real_courses_1]

            accuracy_1 = evaluate(real_courses_1, plan_courses_1.data.cpu().numpy())

            print('Student ' + str(i) + ', Epoch ' + str(epoch) + ': ' + str(loss.data[0]))
            print('enrolled courses in sem 1:', real_courses_1_names)
            print('planned courses in sem 1:', plan_courses_1_names)
            print('Accuracy: ', accuracy_1, '\n')
            if accuracy_1 == 1:
                count_1 += accuracy_1
                with open(target_course+"-top"+str(N), 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow(['real 1:'])
                    writer.writerow(real_courses_1_names)
                    writer.writerow(['plan 1:'])
                    writer.writerow(plan_courses_1_names)
                    writer.writerow([accuracy_1])
                    writer.writerow('----------------------------------------')
                break
            elif old_loss is not None and (old_loss.data[0] - loss.data[0]) / old_loss.data[0] <= 1e-5:
                count_1 += accuracy_1
                with open(target_course+"-top"+str(N), 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow(['real 1:'])
                    writer.writerow(real_courses_1_names)
                    writer.writerow(['plan 1:'])
                    writer.writerow(plan_courses_1_names)
                    writer.writerow([accuracy_1])
                    writer.writerow('-----------------------------------------')
                break
            else:

                old_loss = loss
                loss.backward()
                optimizer.step()
    return count_1/len(student)


if __name__ == '__main__':
    target_course = target_id = sys.argv[1]
    N = int(sys.argv[2])
    f = open('/research/jenny/RNN/data_preprocess/course_id.pkl', 'rb')
    course = pickle.load(f)
    course_id = course['course_id']
    id_course = course['id_course']
    dim_input_course = len(course_id)
    dim_input_grade = len(course_id) * 4
    f = open('/research/jenny/RNN/data_preprocess/grade_id.pkl', 'rb')
    grade = pickle.load(f)
    id_grade = grade['id_grade']

    learning_rate = 1e-3
    loss = "stu" + ".pkl"
    model_name = '/research/jenny/RNN3/train/model1/nw_LSTM_cat_cat_1_50drp0wd1e-05clp0.pkl'
    model1 = torch.load(model_name)
    model1.eval()
    model2 = torch.load(model_name)
    model2.eval()

    eval_set_positive, eval_set_negative = select_evaluation_set(course_id[target_course])
    if eval_set_positive != [] and eval_set_negative != []:
        hit_1 = train(eval_set_positive)
        hit_2 = train(eval_set_negative)
        print(target_course)
        print('positive: ', hit_1)
        print('negative: ', hit_2)
    elif eval_set_positive == []:
        hit_1 = train(eval_set_negative)
        print(target_course)
        print('no positive students')
        print('negative: ', hit_1)
    elif eval_set_negative == []:
        hit_1 = train(eval_set_positive)
        print(target_course)
        print('positive: ', hit_1)
        print('no negative students')



