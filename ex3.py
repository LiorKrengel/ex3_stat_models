import sys
from math import log, pow, exp
from collections import Counter
import os
import re
import numpy as np
import time
import matplotlib.pyplot as plt

develop_txt_name = "develop.txt"
#Getting the parameters
#develop_txt_name = sys.argv[1]
#test_txt_name = sys.argv[2]
#input_word = sys.argv[3]
#output_file_name = sys.argv[4]

# Task defintitions
MIN_WORD_APPEARANCES = 3
NUMBER_OF_CLUSTERS = 9
K = 10
LAMBDA = 0.06 #From assignment 2
EPSILON = 0.001

total_time = time.time()
def get_Data_From_File(file_name):
    ''' 
    Input: file_name the file_name we are reading from
    Output:
        doc_words_counter, tuple contains: cluster_ID, topic_name, document_size, words_counter -> Counter
        words_counter, tuple contains: cluster_ID, topic_name, document_size, words_counter -> Counter
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
                topic_name = match.group(1)#.split('\t')
            if line_number % 4 == 3:
                #We count the number of times each word appeared in this doc
                doc_words_counter = Counter()
                doc_words = line.strip().split()
                for word in doc_words:
                    if word not in doc_words_counter:
                        doc_words_counter[word] = 0

                    if word not in words_counter:
                        words_counter[word] = 0

                    words_counter[word] += 1
                    doc_words_counter[word] += 1
            else:
                continue

            wt = np.zeros(NUMBER_OF_CLUSTERS)
            wt[(topic_ID - 1) % NUMBER_OF_CLUSTERS] = 1
            # Do I need to recalculate the len?
            docs_tuple_list.append((wt, topic_name, len(doc_words), doc_words_counter))
            topic_ID += 1

        #Removing all the words with less then X appearances from all the words
        for word, count in list(words_counter.items()):
            if count <= MIN_WORD_APPEARANCES:
                del words_counter[word]

        #Removing all the words we removed from each doc
        docs_list_filtered = []
        for wt, topic_name, _, doc_words_counter in docs_tuple_list:
            for word in list(doc_words_counter.keys()):
                if word not in words_counter:
                    del doc_words_counter[word]

            docs_list_filtered.append((wt, topic_name, sum([word_cnt for word_cnt in doc_words_counter.values()]), doc_words_counter))

    return docs_list_filtered, words_counter

class EM:
    def __init__(self, docs, words_counter):
        self.docs = docs
        self.words_counter = words_counter
        self.word_count_matrix = np.array([[doc[3].get(word, 0) for word in self.words_counter.keys()] for doc in self.docs])
        self.doc_len_matrix = np.array([doc[2] for doc in self.docs])[:, np.newaxis]  # Matrix of document lengths
        self.P_total = [0 for i in range(NUMBER_OF_CLUSTERS)]
        # Stores the perplexity, likelihood per iteration
        self.iter_scores = []
    
    def run(self):
        t = 1
        while(True):
            if (len(self.iter_scores) > 2 and self.iter_scores[-1][0] - self.iter_scores[-2][0] <= 10):
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
        
    def create_Graph(self, values, graph_name):
        plt.plot(range(1, len(values) + 1), values)
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
            
        print(f"Run of E took:{time.time() - t}")
        return LogL

    def M_Calc(self):
        t = time.time()
        self.set_Alpha_Calc()
        self.P_Calc()
        print(f"Run of M took:{time.time() - t}")
    
    def P_Calc(self):
        t_total = time.time()
        wt_matrix = np.array([doc[0] for doc in self.docs])  # Matrix of weights
        denominator = np.sum(wt_matrix * self.doc_len_matrix, axis=0)
        numerator = np.dot(wt_matrix.T, self.word_count_matrix)
        Pi_matrix = (numerator + LAMBDA) / (denominator[:, np.newaxis] + len(self.words_counter) * LAMBDA)
        self.P_total = {i: {word: Pi_matrix[i, idx] for idx, word in enumerate(self.words_counter.keys())} for i in range(NUMBER_OF_CLUSTERS)}
        print(f"Run of P Calc took: {time.time() - t_total}")

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

if __name__ == '__main__':
    docs, words_counter = get_Data_From_File(develop_txt_name)
    e = EM(docs, words_counter)
    e.run()
    print(f"Total run took:{time.time() - total_time}")