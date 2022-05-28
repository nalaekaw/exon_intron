# Deep neural network for splice sites prediction

Splice site (exon_intron boundaries) prediction is crucial for the analysis of splice isorforms (alternative splicing) of different organisms.

Most of the existing methods achieves a competent performance, but their interpretability remains challenging. Moreover, all traditional machine learning methods manually extracts features, which is tedious job.To address these challenges, a deep learning-based approach is proposed. This approach employs convolutional neural networks (CNN) architecture for automatic feature extraction and effectively predicts splice sites.

### Introduction

The fundamental rule of biology is summarized as a pathway from DNA to RNA
to Protein. This pathway is guided by complex mechanisms transcription (DNA to
RNA) and translation (RNA to protein), respectively.

Most of the protein-coding
genes are interrupted by sequences that do not code for proteins.These intervening
sequences are called Introns and the segments which code for proteins are called
Exons

To produce the matured mRNA, introns must be removed, and its remaining
exons should be spliced together by a process, RNA Splicing. Splice sites bounds
across intron-exon(acceptor sites) and exon-intron(donor sites).

<img src="./images/Снимок экрана от 2022-05-28 16-36-54.png">

<img src="./images/Снимок экрана от 2022-05-28 16-42-50.png">

Identifying splice junctions is one of the critical problems in bioinformatics and
computational biology. It is a binary classification problem which distinguishes true
and false splice sites.

The donor and acceptor splice sites are conserved by di-nucleotide GT/AG respectively in more than 99% of cases.
They
are typically called as candidate splice sites.

Sometimes this orientation can be aligned at different wrong places which makes it a challenging task for the scientist
to identify the exact location of splice sites. However, experimental methods are time consuming and laborious and traditional methods for splice site prediction like alignment-based methods are not reliable. Hence, computational methods are used.

Usually splice site prediction methods falls into three major categories:
- Alignment based methods
- Probabilistic methods
- Machine learning methods

Alignment based methods map compact sequences produced by RNA-sequence to a source genome and determine the splicing sites. However, splice sites reported by alignment are not always accurate, because the risk of randomly mapping a small read to a vast reference genome is high.

Probabilistic methods often depend on the position-specific probabilities of splice sites by computing the likelihood of candidate sites. Many models on this have proposed like GeneSplicer, Markov encoding,random forest,  Bayesian networks, probabilistic SVM.
GeneSplicer combines several methods; they used a
decision tree-based approach called maximal dependence decomposition MDD and
enhanced it with Markov models which catch additional dependencies around splice
site regions. Markov encoding model constitutes of different compositional properties of surrounding splice sites, and the neural networks combine the output from Markov chains to achieve complex communication among nucleotides that improves
detection of splicing junctions. In random forest based approach the splice sites
sequences are aligned position-wise separately using the consensus di-nucleotides,
later this is used to estimate the frequencies and probabilities of nucleotides at
each position. In Bayesian model the dependency graph was constructed to
completely seize the inherent relation between base position in splice site.
Many methods based on Support vector Machine(SVM), Bayesian networks, hidden markov model, Markov Chain Model (MCM) encoding, Hsplice
encoding, Distance Measure (DM) encoding,all these models uses feature combinations built on various statistical strategies to convey splice site characteristics. After feature extraction, various machine learning algorithms such as Support vector machines(SVMs), Random Forest, Naive Bayesian, have practised differentiating real and pseudo splice site. Among all these methods, SVM based approaches has reached an ample position in discriminating true and false sites.

Most of the existing computational methods encounter issues like lack of interpretability, improper separation of feature extraction and model training, manually defined features. These models rely majorly on handcrafted feature set construction. This manual operation for feature representation can be extremely laborious and expensive. The overhead of computational cost limits the dataset size for training the
model, which has a chance of model underfitting. Moreover, concatenating features
sets of different numerical can results in one high dimensional feature space, thereby
difficult to apply machine learning methods

Deep learning, as the modern, leadingedge machine learning method has received ardent audience in the field of artificial
intelligence. It has the ability to automatically extracts features from sequences
and discovers complex representations of data patterns.

Convolutional neural network(CNN) is one of the typical deep learning methods regarded as a cornerstone
in the stream of deep learning models. It has achieved high classification performance in many areas like image processing, natural language processing, speech recognition and also in bioinformatics and computational biology problems.


The stacked architectures employed by CNN uses high-level filters and weight
sharing mechanisms for representing essential features for classification tasks.

CNN has been used in several sequence analysis tasks like:
- predicting transcription factor binding proteins
- promoter non-promoter sequences classification

### Materials and Methods
#### Data collection

Splice sites are categorized as donor and acceptor based upon the consensus dinucleotide GT/AG present at the Exon-Intron and Intron- Exon boundaries respectively. Based upon this distinction, the datasets are also categorized as donor
and acceptor sites. 

```shell
# check connection with google drive
!ls "/content/drive/My Drive/Colab_Notebooks/exon_intron/src/ensembl_seqs/homo_sapiens"
```

```
acceptor_seqs_0_1000_flank_200	     donor_seqs_18000_19000_flank_200
acceptor_seqs_10000_11000_flank_200  donor_seqs_19000_20000_flank_200
acceptor_seqs_1000_2000_flank_200    donor_seqs_2000_3000_flank_200
acceptor_seqs_11000_12000_flank_200  donor_seqs_3000_4000_flank_200
acceptor_seqs_12000_13000_flank_200  donor_seqs_4000_5000_flank_200
acceptor_seqs_13000_14000_flank_200  donor_seqs_5000_6000_flank_200
acceptor_seqs_14000_15000_flank_200  donor_seqs_6000_7000_flank_200
acceptor_seqs_15000_16000_flank_200  donor_seqs_7000_8000_flank_200
acceptor_seqs_16000_17000_flank_200  donor_seqs_8000_9000_flank_200
acceptor_seqs_17000_18000_flank_200  donor_seqs_9000_10000_flank_200
acceptor_seqs_18000_19000_flank_200  other_seqs_0_1000_flank_200
acceptor_seqs_19000_20000_flank_200  other_seqs_10000_11000_flank_200
acceptor_seqs_2000_3000_flank_200    other_seqs_1000_2000_flank_200
acceptor_seqs_3000_4000_flank_200    other_seqs_11000_12000_flank_200
acceptor_seqs_4000_5000_flank_200    other_seqs_12000_13000_flank_200
acceptor_seqs_5000_6000_flank_200    other_seqs_13000_14000_flank_200
acceptor_seqs_6000_7000_flank_200    other_seqs_14000_15000_flank_200
acceptor_seqs_7000_8000_flank_200    other_seqs_15000_16000_flank_200
acceptor_seqs_8000_9000_flank_200    other_seqs_16000_17000_flank_200
acceptor_seqs_9000_10000_flank_200   other_seqs_17000_18000_flank_200
donor_seqs_0_1000_flank_200	     other_seqs_18000_19000_flank_200
donor_seqs_10000_11000_flank_200     other_seqs_19000_20000_flank_200
donor_seqs_1000_2000_flank_200	     other_seqs_2000_3000_flank_200
donor_seqs_11000_12000_flank_200     other_seqs_3000_4000_flank_200
donor_seqs_12000_13000_flank_200     other_seqs_4000_5000_flank_200
donor_seqs_13000_14000_flank_200     other_seqs_5000_6000_flank_200
donor_seqs_14000_15000_flank_200     other_seqs_6000_7000_flank_200
donor_seqs_15000_16000_flank_200     other_seqs_7000_8000_flank_200
donor_seqs_16000_17000_flank_200     other_seqs_8000_9000_flank_200
donor_seqs_17000_18000_flank_200     other_seqs_9000_10000_flank_200
```

To demonstrate method’s generality, six different splice site sequence datasets were collected such as:
- Homo sapiens
- Mus musculus
- Danio rerio
- Drosophila melanogaster
- Arabidopsis thaliana
- C.elegans

The Homo sapiens splice site dataset contains intron, exons splice site
sequences extracted from Esembl


The sequence length of donor and acceptor
sites is 400 nucleotides.

The consent bases GT and AG are present at 200st positions

Final dimension of dataset:
```python
# Dataset dimensions
print(X.shape)
print(Y.shape)
```
```
(812188, 401, 4)
(812188,)
```
#### Convolutional Neural Networks

Convolutional Neural Networks(CNNs) is a specialized form of feed-forward Artificial Neural Networks (ANNs) designed for representing high-level abstraction in data. Convolution is a special kind of linear operation on two functions of realvalued arguments.
CNNs leverages the idea of parameter sharing and sparse connectivity to improve its learning capability.
Generally, a typical CNN consists of three stages,

in the first stage, the layer performs several convolution operations
to produce a set of linear activations.

In the second stage, also called a detector
stage, a non-linear activation like ReLu, sigmoid, tanh are run through the linear
activations.

Finally, a pooling function is used to detect more specific patterns by
reducing the dimensionality throughout the network.

The max-pooling operation reports the maximum value by combining neighborhood activations. One dimensional have become state-of-art performance algorithms in various applications and become feasible due to their simpler and compact configuration.
In proposed model, the simple and compact nature of 1D CNN is leveraged, and
the same is employed for exact splice site classification.
The purpose of using 1D
CNN is to extract the sequential features from the input sequences and to derive
intriguing features from shorter length segments from overall data.


<img src="./images/Снимок экрана от 2022-05-28 21-05-53.png">


The proposed model consists of a series of convolution, pooling (down-sampling),
and fully connected layers, and finally, an output layer. The input to the proposed
architecture is prepared by passing a raw N length long Nucleotide sequence encoded
as N X 4 binary matrix by applying one hot encoding method with A, G, C T as
corresponding columns. The input sequence is a string, as neural networks
work only on numerical data, hence one hot1
encoding technique is used to convert
string to numeric data.

If we consider a sample as a string S = {s1 , s2 , ....sN }
where si ∈ {A,G,C,T}. Each nucleotide of input sequence which is a combination
of this four letters are encoded in to a binary vector with the length of four. They
are represented as (1,0,0,0), (0,1,0,0),(0,0,1,0), and (0,0,0,1) respectively to each
corresponding position of nucleotide {A,G,C,T} as 1, while others as 0.


```python
def one_hot_encode(seq):
    """
    A: [1,0,0,0]
    C: [0,1,0,0]
    G: [0,0,1,0]
    T: [0,0,0,1]
    """

    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')

    return map[np.fromstring(seq, np.int8) % 5]
```

<img src="./images/Снимок экрана от 2022-05-28 21-17-30.png">

#### Performance Measures

<img src="./images/Снимок экрана от 2022-05-28 21-30-56.png">

The performance of model is evaluated by
AUC ROC curve, which said to be commonly used for binary classification problems.

These curves are known to identify the best threshold values for making a
decision.

When dealing with an imbalanced problem,
AUC PR curves gives a more informative picture

<img src="./images/Снимок экрана от 2022-05-28 21-33-27.png">
<img src="./images/Снимок экрана от 2022-05-28 21-34-38.png">
<img src="./images/Снимок экрана от 2022-05-28 21-36-02.png">
<img src="./images/Снимок экрана от 2022-05-28 21-37-21.png">

### Results

The Python functions can be used for doing predictions.

```python
def testing_transcripts(transcript_id, model=model, return_arr=False):
  seq_download = ensRest.getSequenceById(id=transcript_id,expand_5prime=5000, expand_3prime=5000)
  seq_str = seq_download["seq"]
  seq_np_arr = transform_sequence_4_prediction(seq_str)
  #gene_info = get_gene_info(intron_file)
  gene_info, strand = get_intron_delimitation(transcript_id, return_strand=True)
  predictions_seq = model.predict(seq_np_arr)
  arr_0, arr_1,arr2 = sorting_predictions(predictions_seq)
  if strand == -1: 
    print("Intron position predicted by the model: ", get_gene_position_prediction(arr_0, gene_info, donor=False))
    print("Intron position predicted by the model: ", get_gene_position_prediction(arr_1, gene_info))
  else:
    print("Intron position predicted by the model: ", get_gene_position_prediction(arr_0, gene_info))
    print("Intron position predicted by the model: ", get_gene_position_prediction(arr_1, gene_info, donor=False))
  if len(gene_info) == 0:
    print("No introns were found on Ensembl transcript id: ", transcript_id)
    if not return_arr:
      print("Predictions made by exon_intron: ", arr_0, arr_1)
  
  if return_arr:
    return arr_0, arr_1
```

For example, the function testing_transcripts(transcript_id= ‘<Write Gene Ensembl ID>’ model= <Deep Splicer model> ) receives an Ensembl Gene Id as a string and the model as input parameters and does the predictions as shown below:
    
```python
arr_0, arr_1 = testing_transcripts(transcript_id='ENST00000315713', model=model, return_arr=True)
```
```
Intron positions on string:  [(5110, 45127), (45165, 51295), (51395, 54439), (54529, 60286), (60357, 61663), (61762, 62134), (62259, 64779), (64837, 65044)]
Number of sequences predicted per index (donor, acceptor,other):  [   82    92 70071]
Intron position predicted by the model:  ({5110: 2, 45165: 3, 51395: 1, 54529: 11, 60357: 12, 61762: 6, 62259: 0, 64837: 13}, {'top-k': 0.62, 'top-10%': 0.62, 'top-25%': 1.0, 'top-50%': 1.0, 'top-65%': 1.0, 'top-75%': 1.0, 'top-85%': 1.0, 'top-95%': 1.0})
Intron position predicted by the model:  ({45127: 8, 51295: 0, 54439: 16, 60286: 4, 61663: 9, 62134: 1, 64779: 5, 65044: 21}, {'top-k': 0.5, 'top-10%': 0.62, 'top-25%': 1.0, 'top-50%': 1.0, 'top-65%': 1.0, 'top-75%': 1.0, 'top-85%': 1.0, 'top-95%': 1.0})
```
    
### References
    
1. Peren Jerfi CANATALAY, Osman Nuri Ucan. A Bidirectional LSTM-RNN and GRU Method to Exon Prediction Using Splice-Site Mapping. Applied Sciences, 27 April 2022
2. Meher, P.K., Satpathy, S. Improved recognition of splice sites in A. thaliana by incorporating secondary structure information into sequence-derived features: a computational study. 3 Biotech 11, 484 (2021). https://doi.org/10.1007/s13205-021-03036-8
3. Ullah, W., Muhammad, K., Ul Haq, I. et al. Splicing sites prediction of human genome using machine learning techniques. Multimed Tools Appl 80, 30439–30460 (2021). https://doi.org/10.1007/s11042-021-10619-3
4. Conesa, A., Madrigal, P., Tarazona, S. et al. A survey of best practices for RNA-seq data analysis. Genome Biol 17, 13 (2016). https://doi.org/10.1186/s13059-016-0881-8
5. Neelam Goel, Shailendra Singh, Trilok Chand Aseri,
An Improved Method for Splice Site Prediction in DNA Sequences Using Support Vector Machines,
Procedia Computer Science,
Volume 57,
2015,
Pages 358-367,
ISSN 1877-0509,
https://doi.org/10.1016/j.procs.2015.07.350.
6. Kishore Jaganathan, Sofia Kyriazopoulou Panagiotopoulou, Jeremy F. McRae, Siavash Fazel Darbandi, David Knowles, Yang I. Li, Jack A. Kosmicki, Juan Arbelaez, Wenwu Cui, Grace B. Schwartz, Eric D. Chow, Efstathios Kanterakis, Hong Gao, Amirali Kia, Serafim Batzoglou, Stephan J. Sanders, Kyle Kai-How Farh,
Predicting Splicing from Primary Sequence with Deep Learning,
Cell,
Volume 176, Issue 3,
2019,
Pages 535-548.e24,
ISSN 0092-8674,
https://doi.org/10.1016/j.cell.2018.12.015.
7. Kishore Jaganathan, Sofia Kyriazopoulou Panagiotopoulou, Jeremy F. McRae, Serafim Batzoglou. Predicting Splicing from Primary Sequence with Deep Learning. Cell. volume 176, issue 3, P535-548.E24, January 24, 2019.
https://doi.org/10.1016/j.cell.2018.12.015
8. Zhang, Z., Pan, Z., Ying, Y. et al. Deep-learning augmented RNA-seq analysis of transcript splicing. Nat Methods 16, 307–310 (2019). https://doi.org/10.1038/s41592-019-0351-9
9. Daniel Mapleson, Luca Venturini, Gemy Kaithakottil, David Swarbreck, Efficient and accurate detection of splice junctions from RNA-seq with Portcullis, GigaScience, Volume 7, Issue 12, December 2018, giy131, https://doi.org/10.1093/gigascience/giy131
10. Wang, R., Wang, Z., Wang, J. et al. SpliceFinder: ab initio prediction of splice sites using convolutional neural network. BMC Bioinformatics 20, 652 (2019). https://doi.org/10.1186/s12859-019-3306-3
11. V. Akpokiro, O. Oluwadare and J. Kalita, "DeepSplicer: An Improved Method of Splice Sites Prediction using Deep Learning," 2021 20th IEEE International Conference on Machine Learning and Applications (ICMLA), 2021, pp. 606-609, doi: 10.1109/ICMLA52953.2021.00101.
12. Scalzitti, N., Kress, A., Orhand, R. et al. Spliceator: multi-species splice site prediction using convolutional neural networks. BMC Bioinformatics 22, 561 (2021). https://doi.org/10.1186/s12859-021-04471-3
13. Sonnenburg, S., Schweikert, G., Philips, P. et al. Accurate splice site prediction using support vector machines. BMC Bioinformatics 8, S7 (2007). https://doi.org/10.1186/1471-2105-8-S10-S7
14. Baten, A., Chang, B., Halgamuge, S. et al. Splice site identification using probabilistic parameters and SVM classification. BMC Bioinformatics 7, S15 (2006). https://doi.org/10.1186/1471-2105-7-S5-S15
