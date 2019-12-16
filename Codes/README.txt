Replication Codes 
"Mining Narratives From Large Text Corpora" (Elliott Ash, Germain Gauthier, Philine Widmer)
Latest Version: 11/12/2019

The codes should be run in the following order:

************************************
** TRAIN AND TEST TEXT EMBEDDINGS **
************************************

1. make_unigram_sentences_for_nytimes.py 
	- Splits the corpus into sentences
	- Cleans the sentences
	- Writes them in a text file 

2. nytimes_word2vec.py 
	- Trains word embeddings based on the output of make_unigram_sentences_for_nytimes.py
	- We start with pre-trained GloVe vectors, formatted for to the Gensim format: "glove_2_word2vec.6B.300d.txt"

3. nyt_sif_mds_tax.py
	- Computes phrase embeddings based on word embeddings and term frequencies following Arora et al. (2017)

4. check_embeddings_quality.py
	- Helps to check the quality of the phrase embeddings


*****************************
** APPLY SRL TO THE CORPUS **
*****************************

/!\ Version of the AllenNLP's semantic role labeler used: "srl-model-2018.05.25.tar.gz". 
The model should be downloaded from allennlp's archives to run the following codes.
As it is heavy, it is not in this replication folder. /!\

5. nyt_srl_tax_2sents.py
	- Runs the semantic role labeler on the corpus at the sentence level

6. srl_frequencies.py
	- Counts the frequency of each semantic role in the corpus and saves the resulting dictionaries in pickle format.


*********************
** MINE NARRATIVES **
*********************

7. agg_clustering.py
	- Implements hierarchical agglomerative clustering for semantic roles.
	- Outputs dendrograms to graphically represent the clustering procedure.
	- Outputs clusters for each semantic role. The number of clusters should be specified manually.
	- The resulting clusters are written in .csv files for manual inspection.
	- The resulting clusters are written in a dictionary to mine narratives. 
	- Each cluster is labeled by its most frequent term within the corpus.

8. mine_narratives.py 
	- Computes frequencies in large matrices and outputs the most frequent intersections.

*********************
** HYPERPARAMETERS **
*********************

hyperparameters.py is a list of hyperparameters used in the pipeline. For now, we haven't connected it to the rest of the code.


