# latent-Dirichlet-allocation
An implementation of latent Dirichlet allocation (LDA) inference using Gibbs sampling

This project refers to the LDA implementation of the article "Parameter estimation for text analysis", the implicit topic of each word in each document can be inferred using Gibbs sampling. In addition, this project will also output the document-topic counting matrix, topic-word counting matrix and topic counting matrix based on the statistical information of the given dataset.

The basic implementation of the algorithm is in
```
LDA.py
```

If you want to use our project, please add the following code to your program：
```
form LDA import lda
```

To quickly understand the project, we present a simple sample program. Please run
```
example.py
```

The results of the program will be saved in *..\\results\\*, which includes clustering results of given samples, some important distribution of the dataset, and some visualization results.


for any problems, please contact us at cyychenyaoyu@163.com
