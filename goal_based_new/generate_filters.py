__author__ = 'jwj'
import pickle
import pandas as pd
import numpy as np
from utils import *


args = parse_arguments()


class prereqs_pairs():

    def __init__(self):
        super(prereqs_pairs, self).__init__()
        self.data_prereqs = pd.read_csv(args.prereqs_path, header=0)
        self.course = pickle.load(open(args.course_id_path, 'rb'))
        self.course_id = self.course['course_id']
        self.id_course = self.course['id_course']
        self.subject_id = {}
        self.id_subject = {}

    def gene_subject_id_course_subject_dic(self):
        # generate course_id_subject_id_dict
        course_sub = pd.DataFrame({'course_id': list(self.id_course.keys()), 'course': list(self.id_course.values())})
        course_sub['subject'] = course_sub['course'].str.split(' ').str[:-1].apply(' '.join)
        sub = course_sub['subject'].drop_duplicates().reset_index(drop=True)
        self.id_subject = sub.to_dict()

        self.subject_id = dict(zip(self.id_subject.values(), self.id_subject.keys()))
        dic = {'subject_id': self.subject_id, 'id_subject': self.id_subject}
        f = open('subject_id.pkl', 'wb')
        pickle.dump(dic, f)

        course_sub['subject_id'] = course_sub['subject'].apply(lambda x: self.subject_id[x])
        course_id_sub_id = dict(zip(course_sub['course_id'].tolist(), course_sub['subject_id'].tolist()))
        f = open('course_id_sub_id.pkl','wb')
        pickle.dump(course_id_sub_id, f)

    def gene_target_relevant_sub(self):
        self.data_prereqs['target_sub'] = self.data_prereqs['target'].str.split(' ').str[:-1].apply(' '.join).map(self.subject_id)
        self.data_prereqs['prereqs_sub'] = self.data_prereqs['prereqs'].str.split(' ').str[:-1].apply(' '.join).map(self.subject_id)
        self.data_prereqs['prereqs'] = self.data_prereqs['prereqs'].map(self.course_id)
        self.data_prereqs['target'] = self.data_prereqs['target'].map(self.course_id)
        #print(self.data_prereqs)
        self.data_prereqs.dropna(inplace=True)
        #print(len(self.data_prereqs))

        target_sub_prereqs_sub = self.data_prereqs.loc[:, ['target_sub', 'prereqs_sub']].drop_duplicates()
        target_sub_group = target_sub_prereqs_sub.groupby('target_sub')
        dic = {}
        for i in target_sub_group.groups.keys():
            group = target_sub_group.get_group(i)
            dic[i] = set(group['prereqs_sub'].tolist())
        f = open('target_sub_pre_sub_filter.pkl','wb')
        pickle.dump(dic, f)
        f.close()


def gene_sem_courses():
    f = open(args.input_path, 'rb')
    data = pickle.load(f)['stu_sem_grade_condense']
    data = np.array(data)
    num_sem = data.shape[1]
    dic = dict()
    for i in range(num_sem):
        courses_grades = data[:, i]
        courses = []
        for j in courses_grades:
            if j != []:
                courses.extend(np.array(j)[:, 0])
        courses = set(courses)
        dic[i] = courses
    f = open('sem_courses.pkl', 'wb')
    pickle.dump(dic, f)
    f.close()


if __name__ == '__main__':

    prereqs_pair = prereqs_pairs()
    prereqs_pair.gene_subject_id_course_subject_dic()
    prereqs_pair.gene_target_relevant_sub()
    gene_sem_courses()




