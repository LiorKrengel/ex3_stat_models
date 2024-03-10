#   Daniel Bazar    314708181
#   Lior Krengel    315850594
import sys
from math import log, pow, exp
from collections import Counter
import os
import re
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from copy import deepcopy

develop_txt_name = "develop.txt"
topics_file_name = 'topics.txt'

# global parameters defintitions
MIN_WORD_APPEARANCES = 3
NUMBER_OF_CLUSTERS = 9
K = 10
LAMBDA = 0.06
EPSILON = 0.001

total_time = time.time()
def get_Data_From_File(file_name):
    ''' 
    Input: file_name the file_name we are reading from
    Output:
        doc_words_counter, tuple contains: cluster_ID, topics_names, document_size, words_counter -> Counter
        words_counter, tuple contains: cluster_ID, topics_names, document_size, words_counter -> Counter
    '''
    #Each topic is separated by header line, empty line before the text and another empty line
    #So we taking only the third line in each section for the text
    with open(file_name, 'r') as file:   
        topic_pattern = r'\d+\s+(.*?)>'
        topic_ID = 1
        docs_tuple_list = []
        words_counter = Counter()
        for line_number, line in enumerate(file, start =  1):
            # Read the third line and save it
            if line_number % 4 == 1:
                match = re.search(topic_pattern, line)
                topics_names = match.group(1).split()
            elif line_number % 4 == 3:
                #We count the number of times each word appeared in this doc
                doc_words = line.strip().split()
                doc_words_counter = Counter(doc_words)
                words_counter.update(doc_words_counter)

                wt = np.zeros(NUMBER_OF_CLUSTERS)
                wt[(topic_ID - 1) % NUMBER_OF_CLUSTERS] = 1
                docs_tuple_list.append((wt, topics_names, len(doc_words), doc_words_counter))
                topic_ID += 1

        # Removing all the words with less then X appearances from all the words
        words_counter = Counter({k:v for k,v in words_counter.items() if v > MIN_WORD_APPEARANCES})

        #Removing all the words we removed from each doc
        docs_list_filtered = []
        for wt, topics_names, doc_len, doc_words_counter in docs_tuple_list:
            doc_words_counter = Counter({k:v for k,v in doc_words_counter.items() if k in words_counter})
            doc_len = sum(doc_words_counter.values())
            docs_list_filtered.append((wt, topics_names, doc_len, doc_words_counter))

    return docs_list_filtered, words_counter

def get_topics(topics_file=topics_file_name):
    with open(topics_file, 'r') as f:
        file_lines = f.readlines()
    topics = [t.strip() for i,t in enumerate(file_lines) if i%2==0]
    return topics

class EM:
    def __init__(self, docs, words_counter, stopping_criteria=10):
        self.docs = deepcopy(docs)
        self.words_counter = deepcopy(words_counter)
        self.word_count_matrix = np.array([[doc[3].get(word, 0) for word in self.words_counter.keys()] for doc in self.docs])
        self.doc_len_matrix = np.array([doc[2] for doc in self.docs])[:, np.newaxis]  # Matrix of document lengths
        self.P_total = [0 for i in range(NUMBER_OF_CLUSTERS)]
        # Stores the perplexity, likelihood per iteration
        self.iter_scores = []
        self.stopping_criteria = stopping_criteria
    
    def train(self):
        t = 1
        while(True):
            if (len(self.iter_scores) > 2 and self.iter_scores[-1][0] - self.iter_scores[-2][0] <= self.stopping_criteria):
                print("We reached the convergence point")
                break

            self.M_Calc()
            ln_Likelihood = self.E_Calc()
            perplexity = exp(-ln_Likelihood/ sum(self.words_counter.values()))
            self.iter_scores.append((ln_Likelihood, perplexity))
            print(f"LnL: {ln_Likelihood}, Perp: {perplexity}, Iterations: {t}")
            t += 1

        self.create_Graph([likli[0] for likli in self.iter_scores], 'Log Likelihood')
        self.create_Graph([likli[1] for likli in self.iter_scores], 'Perplexity')
        self.create_confusion_matrix_and_fit()
        accuracy = self.accuracy()
        print('Accuracy:', accuracy)


    def create_Graph(self, values, graph_name):
        plt.plot(values)
        plt.xlabel('Iteration')
        plt.ylabel(graph_name)
        plt.title(graph_name)
        plt.savefig(f"{graph_name}.png")
        plt.close()

    def E_Calc(self):
        '''
        E_Calc performing E step and returning the LogLikelihood
        '''
        t = time.time()
        LogL = 0
        for (wt, _, _, doc_words_counter) in self.docs:
            # We will first calculate Zi
            Zi = np.array([log(self.alpha[i]) + sum(word_cnt * log(self.P_total[i][word]) for word, word_cnt in doc_words_counter.items()) for i in range(NUMBER_OF_CLUSTERS)])
            # And we set to be the max Zi
            m = np.max(Zi)
            #E step
            j_indexes = np.where(Zi - m >= -K)[0]
            invalid_indexes = np.where(Zi - m < -K)[0]
            denominator = np.sum(np.exp(Zi[j_indexes] - m))
            wt[invalid_indexes] = 0
            wt[j_indexes] = np.exp(Zi[j_indexes] - m) / denominator
            # Calculate the log-likelihood contribution for this document
            LogL += m + np.log(np.sum(np.exp(Zi - m) * (Zi - m >= -K)))
            
        # print(f"Run of E took:{time.time() - t}")
        return LogL

    def M_Calc(self):
        t = time.time()
        self.set_Alpha_Calc()
        self.P_Calc()
        # print(f"Run of M took:{time.time() - t}")
    
    def P_Calc(self):
        t_total = time.time()
        wt_matrix = np.array([doc[0] for doc in self.docs])  # Matrix of weights
        denominator = np.sum(wt_matrix * self.doc_len_matrix, axis=0)
        numerator = np.dot(wt_matrix.T, self.word_count_matrix)
        Pi_matrix = (numerator + LAMBDA) / (denominator[:, np.newaxis] + len(self.words_counter) * LAMBDA)
        self.P_total = {i: {word: Pi_matrix[i, idx] for idx, word in enumerate(self.words_counter.keys())} for i in range(NUMBER_OF_CLUSTERS)}
        # print(f"Run of P Calc took: {time.time() - t_total}")

    def set_Alpha_Calc(self):
        wt_matrix = np.array([doc[0] for doc in self.docs])  # Matrix of weights
        # Calculate sum of weights for each cluster
        alpha = np.sum(wt_matrix, axis=0) / len(self.docs)
        # Fix for alpha with epsilon
        alpha = np.where(alpha != 0, alpha, EPSILON)
        # Normalize alpha
        alpha_sum = np.sum(alpha)
        alpha /= alpha_sum
        self.alpha = alpha

    def create_confusion_matrix_and_fit(self):
        '''creating confusion matrix and save it as csv.
        in addition, fitting the model with hard clustering. assiging each cluster its predicted topic'''
        topics = get_topics()
        # initialize confusion matrix with zeroes
        confusion_matrix = pd.DataFrame(0, index=range(NUMBER_OF_CLUSTERS), columns=topics)
        for doc in self.docs:
            gold_topics = doc[1]
            predicted_topic = np.argmax(doc[0])
            for t in gold_topics:
                confusion_matrix.iloc[predicted_topic][t] += 1

        self.predicted_topics = confusion_matrix.idxmax(axis=1)
        confusion_matrix['size'] = confusion_matrix.sum(axis=1)
        confusion_matrix.sort_values(by='size', ascending=False, inplace=True)
        confusion_matrix.to_csv('confusion_matrix.csv')

    def accuracy(self):
        '''most run the confusion matrix and fit before'''
        correct = 0
        topics = get_topics()
        for doc in self.docs:
            gold_topics = doc[1]
            predicted_topic_idx = np.argmax(doc[0])
            predicted_topic = self.predicted_topics[predicted_topic_idx]
            if predicted_topic in gold_topics:
                correct+=1
        return correct/len(self.docs)

if __name__ == '__main__':
    docs, words_counter = get_Data_From_File(develop_txt_name)
    print('vocabulary size after filtering:', len(words_counter))
    e = EM(docs, words_counter)
    e.train()
    print(f"Total run took:{time.time() - total_time}")