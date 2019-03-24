import math
import numpy as np
import string

class inverted_file:
    def __init__(self, index_term, posting_list):
        self.index_term = index_term
        self.posting_list = posting_list #store posting

class posting:
    def __init__(self, doc_num, location_list):
        self.doc_num = doc_num
        self.location_list = location_list #store location

def get_keywords():
    keywords, data, n = [], [], 0
    myfile = open('collection-100.txt', 'r')
    for line in myfile.readlines():
        if line != '\n':
            data.append(list())
            words = line.strip().split(' ')
            for word in words:
                trantab = str.maketrans({key: None for key in string.punctuation})
                word = word.translate(trantab).rstrip('s')
                data[n].append(word)
                if len(word) > 3 and word not in keywords:
                    keywords.append(word)
            n += 1
    return keywords, data

def initialize(keywords, data):
    index, unique = [], []
    for _ in range(len(keywords)):
        index.append(inverted_file(keywords[_], []))

    A = np.zeros((len(data), len(keywords)))
    for n in range(len(data)): #n th doc
        unique.append(list())
        for i in range(len(data[n])):#keyword location in doc
            if len(data[n][i]) > 3:
                for j in range(len(keywords)):#j th keyword
                    if data[n][i].find(keywords[j]) != -1:#match this word with keyword
                        A[n][j] += 1
                        #update index
                        _ = True
                        for m in range(len(index[j].posting_list)):#go throuth all posting
                            if index[j].posting_list[m].doc_num == n:
                                index[j].posting_list[m].location_list.append(i)
                                _ = False
                                break
                        if _:#no such posting, append it
                            index[j].posting_list.append(posting(n,[i]))
    
    for k in range(A.shape[1]):#k th keyword
        frequency = A[:,k]
        t = np.where(frequency > 0)
        if len(t[0]) == 1:
            unique[t[0][0]].append(keywords[k])
    
    W = np.zeros((len(data), len(keywords)))#get weight matrix
    for i in range(W.shape[0]):#i th doc
        for j in range(W.shape[1]):# j th keyword
            W[i][j] = math.log(float(len(data))/len(A[:, j].nonzero()), 2)*A[i][j]/A[i, :].max()
    return A, W, index, unique

def get_vector(query):
    w = np.zeros(len(keywords))
    words = query.split(' ')
    for i in range(len(words)):
        trantab = str.maketrans({key: None for key in string.punctuation})
        words[i] = words[i].translate(trantab).rstrip('s')
        if len(words[i]) > 3:
            for j in range(len(keywords)):
                if keywords[j].find(words[i]) != -1:
                    w[j] += 1
    return w.T

def display(A, W, query, index, unique):
    angles = np.matmul(W, query)
    max_angle_index, max_angle_value = np.zeros(3), np.zeros(3)
    for i in range(len(angles)):#i th doc
        angles[i] = angles[i]/(np.sqrt(W[i].dot(W[i]))+np.sqrt(query.dot(query)))
        for j in range(3):
            #mark top3 doc
            if angles[i] > max_angle_value[j]:
                max_angle_value[j] = angles[i]
                max_angle_index[j] = i
                break
    
    max_index, max_value = np.zeros((3,5)), np.zeros((3,5))
    for i in range(3):#1-3doc
        for j in range(W.shape[1]):#all keywords' weight
            for k in range(5):#mark top5 keywords
                if W[int(max_angle_index[i]), j] > max_value[i, k]:
                    max_value[i, k] = W[int(max_angle_index[i]), j]
                    max_index[i, k] = j
                    break

    #display 3*top5 keywords
    for i in range(3):
        print(int(max_angle_index[i]+1))
        for j in range(5):
            print(index[int(max_index[i, j])].index_term, end = ' -> | ')
            for posting in index[int(max_index[i, j])].posting_list:
                print('D' + str(posting.doc_num+1), end = ':')
                for location in posting.location_list:
                    print(str(location), end =',')
                print(' |' , end ='')
            print('\n')
        print(len(unique[int(max_angle_index[i])]))#Number of unique keywords in document
        print(np.sqrt(A[int(max_angle_index[i]),:].dot(A[int(max_angle_index[i]),:])))#Magnitude of the document vector
        print(max_angle_value[i])#Similarity score
        print('\n')

keywords, data = get_keywords()
A, W, index, unique = initialize(keywords, data)
while(1):
    query = get_vector(input("input:"))
    display(A, W, query, index, unique)