# -*- coding: utf-8 -*-
"""
Examples of the latent Dirichlet allocation using Gibbs sampling

This script runs an example that demonstrate the output of LDA on a simple data.

@codeauthor: CyyJenkins
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from LDA import lda
from openpyxl import Workbook

       
def distributionmap(results, category_names):
    """ Discrete distribution as horizontal bar chart
    
    Parameters
    ----------
    results (dict): 
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
        
    category_names (list of str): The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=11)
    plt.show()

    return fig, ax


def heatmap(dis_data, col_name, row_name):
    """ Heatmap plotting
    
    Parameters
    ----------    
    dis_data (numpy.array): (Type_of_event x type_of_attributes) the counting array
    col_name (list of str): Name of different type of events. In our experiment
                            it represents different document names
    row_name (list of str): Name of different type of attributes. In our 
                            experiment it represents different topic names
    """
    
    fig, ax = plt.subplots()
    im = ax.imshow(dis_data)
    
    ax.set_xticks(np.arange(len(row_name)))
    ax.set_yticks(np.arange(len(col_name)))
    ax.set_xticklabels(row_name)
    ax.set_yticklabels(col_name)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")    
    for i in range(len(col_name)):
        for j in range(len(row_name)):
            text = ax.text(j, i, dis_data[i, j],
                           ha="center", va="center", color="w")
    
    ax.set_title("Document-topic distribution")
    fig.tight_layout()
    plt.show()


def xlsx_save(data,path):
    wb = Workbook()    
    ws = wb.active   
    [h, l] = data.shape  # h:rows，l:columns
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i,j])
        ws.append(row)
    wb.save(path)
    


if __name__ == '__main__':
    
    time_start = time.perf_counter()
    
    # data reading
    data = []
    file = open('data.txt','r')
    for line in file:
        data.append(np.array(list(map(int, line.split()))))
    file.close()
    
    
    # parameter setting
    iter_num = 100
    word_num = 10
    topic_num = 4
    alpha, beta = 1, 1
    
    
    # lda infering
    docs = lda(data, topic_num, iter_num, word_num, alpha=alpha, beta=beta)
    (n_doc_topic, n_topic_word, n_topic, doc_topic, liks) = docs.lda_infer() 
    
        
    # figure plotting
    topic_name = ['topic_%s'%(_+1) for _ in range(topic_num)]
    doc_name = ['document_%s'%(_+1) for _ in range(len(data))]
    results = {}
    for i in range(len(data)):
        results[doc_name[i]] = n_doc_topic[i,:]
        
    distributionmap(results, topic_name)    
    heatmap(n_doc_topic, doc_name, topic_name)
    
    
    # data saving
    print('Start data saving...')
    xlsx_save(n_doc_topic, 'results\\doc_topic.xlsx')
    xlsx_save(n_topic, 'results\\topic_dis.xlsx')
    xlsx_save(n_topic_word, 'results\\topic_word.xlsx')
    word_topic = np.array([j for i in doc_topic for j in i])
    xlsx_save(word_topic[:, np.newaxis], 'results\\word_topic.xlsx')
    
    
    running_time = (time.perf_counter() - time_start)
    print("Time used: %.6ss"%(running_time))

    