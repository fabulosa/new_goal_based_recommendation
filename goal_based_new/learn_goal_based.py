__author__ = 'jwj'
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import pickle
import pandas as pd


class goal_based_rec(nn.Module):

    def __init__(self, ckt_model1, ckt_model2, course_id_path, dim_input_course, dim_input_grade, target_course):  # number of planning semesters == 3
        super(goal_based_rec, self).__init__()

        self.target_course = target_course
        split = self.target_course.split(' ')
        self.t_num = int(''.join(list(filter(str.isdigit, split[-1]))))
        self.t_sub = ' '.join(split[:-1])
        self.dim_input_course = dim_input_course
        self.dim_input_grade = dim_input_grade

        self.past_model = ckt_model1  # the trained grade prediction model
        self.plan_model_1 = ckt_model2

        self.past_model.eval()
        self.plan_model_1.eval()

        # default parameters
        self.plan_input_1 = nn.parameter.Parameter(torch.ones(dim_input_course+dim_input_grade), requires_grad=True)

        # recommended courses (top 10)
        self.plan_courses_1 = Variable(torch.ones(dim_input_course+dim_input_grade).unsqueeze(0).unsqueeze(0), requires_grad=True)

        # outputs in past and planning
        self.past_output = None

        self.target_prereqs_sub_filter = pickle.load(open('target_sub_pre_sub_filter.pkl', 'rb'))
        subject = pickle.load(open('subject_id.pkl', 'rb'))
        self.subject_id = subject['subject_id']
        self.id_subject = subject['id_subject']
        self.course_id_sub_id = pickle.load(open('course_id_sub_id.pkl', 'rb'))
        f = open(course_id_path, 'rb')
        course = pickle.load(f)
        self.course_id = course['course_id']
        self.id_course = course['id_course']

        f = open('sem_courses.pkl', 'rb')
        self.sem_courses = pickle.load(f)

    def forward(self, padded_input, seq_len, courses_not_taken, k):  # personalized to one student, so no batch; the last recommended semester is k

        hidden_a = torch.zeros(1, 1, 50).cuda()
        hidden_b = torch.zeros(1, 1, 50).cuda()
        initial_hidden = [Variable(hidden_a), Variable(hidden_b)]

        print("feed histories")
        hidden_past, self.past_output = self.cal_hiddenstates_output(padded_input, seq_len, courses_not_taken, k-1, self.past_model, initial_hidden)
        self.past_output = Variable(self.real_output(self.past_output), requires_grad=False).cuda()
        self.plan_courses_1 = self.plan_input_1.unsqueeze(0).unsqueeze(0).cuda() * self.past_output.unsqueeze(0).unsqueeze(0)
        #self.plan_courses_1 = self.round_to_01(self.plan_courses_1)

        print("feed the first semester")
        self.plan_model_1.hidden = hidden_past
        self.plan_model_1.batch_size = 1
        output = self.plan_model_1(self.plan_courses_1, [1])

        return output

    # construct output by candidate courses
    def real_output(self, output):  # flag: detect whether it's the output for rec courses for semester 1 or 2
        candidate_course = torch.zeros(self.dim_input_course)
        # course part only has target course == 1
        candidate_course[self.course_id[self.target_course]] = 1
        candidate_course_grade = torch.zeros(self.dim_input_grade)
        index = list(map(int, np.array(output)*4))
        candidate_course_grade[index] = 1
        real_output = torch.cat((candidate_course, candidate_course_grade)).cuda()
        return real_output

    def level(self, num):
        if num <= 99:
            level = 0
        elif num <= 199:
            level = 1
        else:
            level = 2
        return level

    def filter_course_level(self, target_course_level):  # only leave courses not higher the than target course
        course_pd = pd.DataFrame(self.id_course.items(), columns=['id', 'name'])
        course_pd['num'] = course_pd['name'].apply(lambda x: int(''.join(list(filter(str.isdigit, x.split(' ')[0])))))
        course_pd['level'] = course_pd['num'].apply(lambda x: self.level(x))
        courses = course_pd.loc[course_pd['level'] <= target_course_level]['id'].tolist()
        return courses

    # calculate hidden states and candidate courses to recommend for the next semester
    def cal_hiddenstates_output(self, padded_input, seq_len, courses_not_taken, k, model, hidden):  # hidden: initial hidden states
        # filter 1: only consider courses available in that semester
        rec_courses1 = self.sem_courses[k]

        # filter 2: consider courses excluding target course
        rec_courses2 = set(self.course_id.values()) - set([self.course_id[self.target_course]])

        model.eval()
        # apply pre_calculated hidden states
        model.batch_size = 1
        model.hidden = hidden
        # compute output
        y_pred = model(padded_input, seq_len)

        y_last_period = y_pred[0, -1]
        y_last_period = y_last_period.contiguous().view(self.dim_input_course, 4)
        y_B_exp = torch.exp(y_last_period[:, :2])
        y_B = y_B_exp / torch.sum(y_B_exp, dim=1)[:, None]
        y_B = y_B.data[:, 0]

        # filter 3: only consider courses with grade predicted > B
        rec_courses3 = np.where(y_B.cpu() > 0.5)[0]

        # filter 4: consider courses student has not taken
        rec_courses4 = courses_not_taken

        # filter 5: don't consider courses in higher levels than the target course
        rec_courses5 = self.filter_course_level(self.level(self.t_num))

        # filter 6: subject filter, only consider courses in the same subject as the target course's subject, and courses in those subjects that have been the prerequisite courses for the target courses
        related_sub = self.target_prereqs_sub_filter[self.subject_id[self.t_sub]]
        rec_courses6 = [j for j in self.course_id.values() if self.course_id_sub_id[j] in related_sub]
        candidates = set.intersection(set(rec_courses1), set(rec_courses2), set(rec_courses3), set(rec_courses4), set(rec_courses5), set(rec_courses6))
        if candidates == set():  # if no intersection, loose subject filter
            candidates = set.intersection(set(rec_courses1), set(rec_courses2), set(rec_courses3), set(rec_courses4), set(rec_courses5))

        new_hidden = model.hidden

        candidates = list(map(int, candidates))
        #print('final', [self.id_course[i] for i in candidates])
        return new_hidden, list(candidates)

    # maximize the probability of getting above threshold for the target course
    def loss(self, output):

        label = Variable(torch.LongTensor([0]), requires_grad=False).cuda()
        output_target = output.view(-1, 4)[self.course_id[self.target_course], :2].unsqueeze(0)
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(output_target, label)
        return loss
