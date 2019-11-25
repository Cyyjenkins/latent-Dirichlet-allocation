# -*- coding: utf-8 -*-
"""
An implementation of Latent Dirichlet Allocation

This project refers to the LDA implementation of the article "Parameter 
estimation for text analysis", the implicit topic of each word in each document
can be inferred using Gibbs sampling. In addition, this project will also 
output the document-topic counting matrix, topic-word counting matrix and 
topic counting matrix based on the statistical information of the given dataset.

@codeauthor: CyyJenkins
"""

import copy
import pylab
import random
import numpy as np
 

class lda(object):
    """
    Latent Dirichlet Allocation   
    
    Parameters
    ----------
        data (list): (document_num x document_len) Stores the words that appear
                     in sequence in each document, each word represented by an 
                     integer number
        k          (int): number of topics
        iterations (int): iteration number for lda inferring step
        num_word   (int): Number of word types
        alpha (int) and beta (int): initial value of hyper-parameter of 
                                    Beta distribution, usually set as int                                    
    Raises:
        Exception: If the any of the inputs are an incorrect type.
    """
    
    
    def __init__(self, data, k, iterations, num_word, alpha=1, beta=1): 
        
        # Raise Exception if the any of the inputs are an incorrect type
        if not (type(data)==list) & (len(data)>1):
            raise Exception('Please enter the expected type of input')
            
        if not (type(k)==int) & (type(iterations)==int) & (type(num_word)==int):
            raise Exception('Please enter the model parameters as int')
            
        if not (type(alpha)==int) & (type(beta)==int):
            raise Exception('Please enter the hyper-parameters as int')
               
        # Store input data and model parameters                      
        self.__data = data 
        self.__num_doc = len(data)
        self.__num_word = num_word
        self.__k = k
        self.__iter = iterations
        
        # Store hyper-parameters of Beta distribution
        self.__alpha = alpha
        self.__beta = beta
        
        # Calculate the frequency of occurrence of each word in each document
        # And store it as a matrix (document x word)
        dw_count = np.zeros((len(data), num_word), dtype=int) # doc_word_count
        for i in range(len(data)):
            for j in range(data[i].shape[0]):
                dw_count[i,data[i][j]] += 1
        self.__doc_word = dw_count
       
    
    def number_account(self, doc_word, doc_topic): 
        """Calculate the initial model distribution based on randomly
        generated topics before using Gibbs sampling. 
        
        Parameters
        ----------     
        n_doc_top   (numpy.array): (doc_num x topic_num) number of topics for each document
        n_topic_word(numpy.array): (topic_num x word_num) number of words for each topic
        n_topic     (numpy.array): (topic_num x 1) number of topics        
        """
        
        # Matrix creation
        n_doc_topic = np.zeros((self.__num_doc, self.__k), dtype=int)
        n_topic_word = np.zeros((self.__k, self.__num_word), dtype=int)
        n_topic  = np.zeros((self.__k, 1), dtype=int)
        
        # Frequency calculation
        for d in range(self.__num_doc):
            N = len(doc_word[d])
            for n in range(N):
                w = doc_word[d][n]
                z = doc_topic[d][n]
                n_doc_topic[d][z] += 1
                n_topic_word[z][w] += 1
                n_topic[z] += 1
        
        return n_doc_topic, n_topic_word, n_topic
    
    
    def multinomial_sampling(self, d, w, n_doc_topic, n_topic_word, n_topic):
        """Inferring the hidden state (topic) of each word of each document
        using Gibbs sampling, the sampling probability of each word of each 
        document are based on its conditional probability         
        """
        
        # Sampling probability 
        P = [ 0.0 ] * self.__k
        P = ((n_doc_topic[d,:] + self.__alpha )*(n_topic_word[:,w] + self.__beta)).T \
        /(n_topic[:] + self.__num_word*self.__beta).reshape(self.__k,)
        
        # Convert probability density function into cumulative distribution function
        for z in range(1,self.__k):
            P[z] = P[z] + P[z-1]
        
        # Topic sampling
        rnd = P[self.__k - 1] * random.random()
        for z in range(self.__k):
            if P[z] >= rnd:
                return z
    
    
    def likelihood_calculation(self, n_doc_topic, n_topic_word, n_topic):
        """Calculate the log likelihood based on current document-topic 
        distribution and topic-word distribution.
        
        The log-likelihood of each observed word can be calculated by
        accumulating the probability of generation of the word under different
        topics. In this case, the log likelihood of the model can be obtained
        by log likelihood multiplication of words from all of documents.        
        """
                
        lik = 0
        
        # The probability of each word under each topic
        Pw_z = (n_topic_word.T + self.__beta) / (n_topic.T + self.__num_word *self.__beta)
                
        for d in range(len(self.__data)):
            # The probability of each topic
            Pz = (n_doc_topic[d] + self.__alpha) / (np.sum(n_doc_topic[d]) \
                  + self.__k*self.__alpha) 
            
            # Joint probability of each word and its different topics
            Pwz = Pz * Pw_z
            # The probability of each word in the current situation
            Pw = np.sum( Pwz , 1 ) + 1e-6            
            # Notice that the multiplication of the probability becomes
            # cumulative under logarithmic conditions
            lik += np.sum( self.__doc_word[d] * np.log(Pw) )    
            
        return lik
    
    
    def plot(self, it, liks):
        """Update the log-likelihood figure of the model after each iteration"""
               
        print('iterations:%s   log-likelihood:%.8s'%(it, liks[-1]))
    
        pylab.clf()
        pylab.title( "log-likelihood of LDA clustering" )
        pylab.xlabel('iterations')
        pylab.ylabel('log-likelihood')
        
#        pylab.plot( np.arange(1,1+len(liks)).astype(dtype=np.str) , liks)
        pylab.plot(np.arange(1,1+len(liks),1), liks)
        pylab.draw()
        pylab.pause(1e-4)
    
    
    def lda_infer(self):
        """Calculate document-topic distribution, topic-word distribution, 
        topic distribution, and infer the implicit topic of all words appearing
        in the document by Gibbs sampling.

        The inference process for LDA using Gibbs sampling is:
            1 Randomly initialize the implied theme for the words in the input document
            2 Calculate each random distribution parameter according to the
              document-word and initialized subject
            3 In a given number of iterations, calculate:
                3.1 Select\Replace a certain type of word under a document, and 
                    reduce the number of corresponding positions of the 
                    document-word counting matrix by one.
                3.2 Obtain new topics sampled by edge distribution probability
                3.3 Add the number of corresponding positions of the 
                    document-word counting matrix
                3.4 Completing a round of document-word sampling, 
                    recalculating the model log likelihood
            4 Output parameters:
                Document-topic counting matrix, subject-word counting matrix, 
                subject counting matrix, subject of each word under each 
                document, pattern log likelihood for each iteration     
        """
        
        print('Start LDA inferring...')
        pylab.ion()
        liks = []        
        
        # Create the document_word matrix and document_topic matrix
        doc_word = copy.deepcopy(self.__data)
        doc_topic = [ None for i in range(self.__num_doc) ]
        
        # Randomly initialize the implied theme for the words in the input document
        for d in range(self.__num_doc):
            doc_topic[d] = np.random.randint(0, self.__k, len(doc_word[d])).tolist()
        
        # Document_topic counting matrix, topic_word counting matrix
        # and topic counting matrix
        (n_doc_topic, n_topic_word, n_topic) = self.number_account(doc_word, doc_topic)
        
        print('Start iterations...')        
        for it in range(self.__iter):        # for each iterations
            for d in range(self.__num_doc):  # for each document
                N = len(doc_word[d])
                for n in range(N):           # for each type of word
                    w = doc_word[d][n]
                    z = doc_topic[d][n]            
        
                    # reduce the counting matrix by one.
                    n_doc_topic[d][z] -= 1
                    n_topic_word[z][w] -= 1
                    n_topic[z] -= 1
                    
                    # sampling according to coditional probability
                    z = self.multinomial_sampling(d, w, n_doc_topic, n_topic_word, n_topic)
        
                    # Save the topic of target word and add the counting matrix by one.
                    doc_topic[d][n] = z
                    n_doc_topic[d][z] += 1
                    n_topic_word[z][w] += 1
                    n_topic[z] += 1
         
            # log-likelihood
            lik = 0
            lik = self.likelihood_calculation(n_doc_topic, n_topic_word, n_topic)
            liks.append(lik)
            self.plot(it, liks)
            
        pylab.ioff()
        pylab.tight_layout()
        pylab.show() 
        print('LDA clustering complete...')
        return n_doc_topic, n_topic_word, n_topic, doc_topic, liks
