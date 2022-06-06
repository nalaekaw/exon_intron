# Deep neural network for splice sites prediction

## Introduction 

The fundamental rule of biology is summarized as a pathway from DNA to RNA to Protein. This pathway is guided by complex mechanisms transcription (DNA to RNA) and translation (RNA to protein), respectively. Most of the protein-coding genes are interrupted by sequences that do not code for proteins. These intervening sequences are called Introns and the segments which code for proteins are called Exons. To produce the matured mRNA, introns must be removed, and its remaining exons should be spliced together by a process, RNA Splicing. Splice sites bounds across intron-exon (acceptor sites) and exon-intron (donor sites). 

Identifying splice junctions is one of the critical problems in bioinformatics and computational biology. Splice site (exon-intron boundaries) prediction is crucial for the analysis of splice isoforms (alternative splicing) of different organisms. 

It is a binary classification problem which distinguishes true and false splice sites. Exact splice site classification is necessary to locate genes in nucleotide sequence and to outline gene structure (Sonnenburg et al.,2002). The donor and acceptor splice sites are conserved by di-nucleotide GT/AG respectively in more than 99% of cases. They are typically called as candidate splice sites (Sheth et al,.2006). Sometimes this orientation can be aligned at different wrong places which makes it a challenging task for the scientist to identify the exact location of splice sites. However, experimental methods are time consuming and laborious and traditional methods for splice site prediction like alignment-based methods are not reliable. Hence, computational methods are used.  Usually splice site prediction methods falls into three major categories:
- Alignment based methods
- Probabilistic methods
- Machine learning methods. 

<img src="./images/Снимок экрана от 2022-05-28 16-36-54.png">

<img src="./images/Снимок экрана от 2022-05-28 16-42-50.png">

Splice site prediction is crucial for understanding underlying gene regulation, gene function for better genome annotation. Many computational methods exist for recognizing the splice sites. Although most of the methods achieves a competent performance, their interpretability remains challenging. Moreover, all traditional machine learning methods manually extracts features, which is tedious job. 

Based on the foregoing, the aim of this work is to propose a new computational approach to address existing challenges. 

To achieve this goal, the following tasks were set: 

1. Collect data for model training and testing 

2. Implement deep learning-based approach that employs convolutional neural networks (CNN) architecture for automatic feature extraction and effectively predicts splice sites. 

3.  Evaluate model using the human genome 

4. Evaluate using other species’ genomes to identify how well a model trained on the human genome can generalize to other species since the objective is to use this model as a first-line annotation tool for newly sequenced genome

## 1 Literature review 

### 1.1 Alignment based methods 

Alignment based methods map compact sequences produced by RNA-sequence to a source genome and determine the splicing sites. However, splice sites reported by alignment are not always accurate, because the risk of randomly mapping a small read to a vast reference genome is high. Some of the procedures for recognising splicing from RNA-sequence data is by the gapped arrangement of RNA-seq reads to the source genome. Some mapping strategies require structural annotation of exon coordinates, many recently developed algorithms can perform ab initio alignment, and are independent of annotation, besides, can possibly recognize unknown splice junctions by the evidence of spliced alignments. 

HISAT (hierarchical indexing for spliced alignment of transcripts) is a highly efficient system for aligning reads from RNA sequencing experiments. HISAT uses an indexing scheme based on the Burrows-Wheeler transform and the Ferragina-Manzini (FM) index, employing two types of indexes for alignment: a whole-genome FM index to anchor each alignment and numerous local FM indexes for very rapid extensions of these alignments. HISAT’s hierarchical index for the human genome contains 48,000 local FM indexes, each representing a genomic region of ~64,000 bp. Tests on real and simulated data sets showed that HISAT is the fastest system currently available, with equal or better accuracy than any other method. Despite its large number of indexes, HISAT requires only 4.3 gigabytes of memory. HISAT supports genomes of any size, including those larger than 4 billion bases (Kim et al.,2015). 

Mouse transcriptomes were mapped and quantified by deeply sequencing them and recording how frequently each gene is represented in the sequence sample (RNA-Seq). This provides a digital measure of the presence and prevalence of transcripts from known and previously unknown genes. Reference measurements were reported. It composed of 41–52 million mapped 25-base-pair reads for poly(A)-selected RNA from adult mouse brain, liver and skeletal muscle tissues.  RNA standards were used to quantify transcript prevalence and to test the linear range of transcript detection, which spanned five orders of magnitude. Although >90% of uniquely mapped reads fell within known exons, the remaining data suggest new and revised gene models, including changed or additional promoters, exons and 3′ untranscribed regions, as well as new candidate microRNA precursors. RNA splice events, which are not readily measured by standard gene expression microarray or serial analysis of gene expression methods, were detected directly by mapping splice-crossing sequence reads. Distinct splices 1.45 × 105 were observed, and alternative splices were prominent, with 3,500 different genes expressing one or more alternate internal splices (Mortazavi et al., 2008). 

In comparison testing, GSNAP has speeds comparable to existing programs, especially in reads of ≥70 nt and is fastest in detecting complex variants with four or more mismatches or insertions of 1–9 nt and deletions of 1–30 nt. Although SNP tolerance does not increase alignment yield substantially, it affects alignment results in 7–8% of transcriptional reads, typically by revealing alternate genomic mappings for a read. Simulations of bisulfiteconverted DNA show a decrease in identifying genomic positions uniquely in 6% of 36 nt reads and 3% of 70 nt reads (Wu et al., 2010). 

The accurate mapping of reads that span splice junctions is a critical component of all analytic techniques that work with RNA-seq data. A second generation splice detection algorithm was introduced, MapSplice, whose focus is high sensitivity and specificity in the detection of splices as well as CPU and memory efficiency. MapSplice can be applied to both short (<75 bp) and long reads (75 bp). MapSplice is not dependent on splice site features or intron length, consequently it can detect novel canonical as well as non-canonical splices. MapSplice leverages the quality and diversity of read alignments of a given splice to increase accuracy. MapSplice achieves higher sensitivity and specificity than TopHat and SpliceMap on a set of simulated RNA-seq data. Experimental studies also support the accuracy of the algorithm. Splice junctions derived from eight breast cancer RNA-seq datasets recapitulated the extensiveness of alternative splicing on a global level as well as the differences between molecular subtypes of breast cancer. These combined results indicate that MapSplice is a highly accurate algorithm for the alignment of RNA-seq reads to splice junctions. Software download URL: http://www.netlab.uky .edu/p/bioinfo/MapSplice (Wang et al., 2010). 

A “big data” approach was developed that predicts novel splice junctions derived from RNA-seq data. A deep convolutional neural network-based methodology was adopted to predict splice junctions based on the splicing signals present in their donor and acceptor flanking sequences. The approach is independent of the read support and reoccurrence derived from RNA-seq. Approach was applied to the GENCODE project gene annotation data and found that deep learning outperforms other state-of-the-art approaches based on testing of experimental dataset and comparison of evaluation metrics. In experiments, DeepSplice achieved approximately 96% accuracy, almost 30% higher than conventional models of predicting splice sites. In addition, this big data approach is particularly efficient and scalable when classifying a large amount of splice junctions by deploying the computation task to multiple CPUs and GPUs, 350 junctions per second of CPU time. (Yi Zhang et al., 2016). 

 

### 1.2 Probabilistic methods 

Probabilistic methods often depend on the position-specific probabilities of splice sites by computing the likelihood of candidate sites. Many models on this have proposed like GeneSplicer, Markov encoding, random forest, Bayesian networks, probabilistic SVM. GeneSplicer combines several methods; they used a decision tree-based approach called maximal dependence decomposition MDD and enhanced it with Markov models which catch additional dependencies around splice site regions. Markov encoding model constitutes of different compositional properties of surrounding splice sites, and the neural networks combine the output from Markov chains to achieve complex communication among nucleotides that improves detection of splicing junctions. In random forest based approach the splice sites sequences are aligned position-wise separately using the consensus di-nucleotides, later this is used to estimate the frequencies and probabilities of nucleotides at each position. In Bayesian model Chen TM, constructed the dependency graph to completely seize the inherent relation between base position in splice site. 

A Markov/neural hybrid approach was presented to detect signals, such as SS, TSS, and TIS, in genomic sequences. The approach used Markov encoding: The inputs to the neural networks are Markovian probabilities that characterize the prior biological knowledge on coding and noncoding regions. The neural networks capture intrinsic features surrounding the signals by combining lower-order Markovian probabilities and finding an appropriate arbitrary mapping that represents the signal accurately. As shown, the present hybrid model effectively implements a complex higher-order Markov model. Although the higher-order Markov models were often touted earlier as accurate models to characterize signals, their direct implementation has been practically prohibitive because of the need for estimating the large number of parameters with the often-limited amount of training data (Rajapakse et al., 2005). 

Paper presents a novel approach for donor splice site prediction that involves three splice site encoding procedures and application of RF methodology. The proposed approach discriminated the TSS from FSS with higher accuracy. Also, the RF outperformed SVM, ANN, Bagging, Boosting, Logistic regression, kNN and Naïve Bayes classifiers in terms of prediction accuracy. Further, RF with the proposed encoding procedures showed high prediction accuracy both in balanced and imbalanced situations. Being a supplement to the commonly used ss prediction methods, the proposed approach is believed to contribute to the prediction of eukaryotic gene structure. The web server will help the user for easy prediction of donor ss (Meher et al., 2016). 

A dependency graph model was developed to fully capture the intrinsic interdependency between base positions in a splice site. The establishment of dependency between two position is based on a χ2-test from known sample data. To facilitate statistical inference, the dependency graph was expanded  (which is usually a graph with cycles that make probabilistic reasoning very difficult, if not impossible) into a Bayesian network (which is a directed acyclic graph that facilitates statistical reasoning). When compared with the existing models such as weight matrix model, weight array model, maximal dependence decomposition, tree model as well as the lessstudied second-order and third-order Markov chain models, the expanded Bayesian networks from dependency graph models perform the best in nearly all the cases studied (Chen et al., 2004). 

The method for splice site detection was proposed. It consists of two stages: a first order Markov model (MM1) is used in the first stage and a support vector machine (SVM) with polynomial kernel is used in the second stage. The MM1 serves as a pre-processing step for the SVM and takes DNA sequences as its input. It models the compositional features and dependencies of nucleotides in terms of probabilistic parameters around splice site regions. The probabilistic parameters are then fed into the SVM, which combines them nonlinearly to predict splice sites. When the proposed MM1-SVM model is compared with other existing standard splice site detection methods, it shows a superior performance in all the cases (Baten et al., 2006). 

Many methods based on Support vector Machine (SVM) ,Bayesian networks, hidden markov model, Markov Chain Model (MCM) encoding, Hsplice encoding, Distance Measure (DM) encoding, all these models use feature combinations built on various statistical strategies to convey splice site characteristics. After feature extraction, various machine learning algorithms such as Support vector machines (SVMs), Random Forest, Naive Bayesian, have practised differentiating real and pseudo splice site. Among all these methods, SVM based approaches has reached an ample position in discriminating true and false sites. K.Meher employed a computational approach and extracted features based on positional and compositional dependency and sent them into SVM for classification. Zhang used linear SVM with Bayes kernel to distinguish consensus di-nucleotide GT/AG. In another approach, prediction is done by using orthogonal encoding through SVM. 

 

Each splice site sequence was transformed into a numeric vector of length 49, out of which four were positional, four were dependency and 41 were compositional features. Using the transformed vectors as input, prediction was made through support vector machine. Using balanced training set, the proposed approach achieved area under ROC curve (AUC-ROC) of 96.05, 96.96, 96.95, 96.24 % and area under PR curve (AUC-PR) of 97.64, 97.89, 97.91, 97.90 %, while tested on human, cattle, fish and worm datasets respectively. On the other hand, AUC-ROC of 97.21, 97.45, 97.41, 98.06 % and AUC-PR of 93.24, 93.34, 93.38, 92.29 % were obtained, while imbalanced training datasets were used. The proposed approach was found comparable with state-of-art splice site prediction approaches, while compared using the bench mark NN269 dataset and other datasets (Meher et al., 2016).  

In paper was presented a novel idea of constructing a mapping method from Bayes’ rule. This mapping method was then integrated with SVM classifier and applied to the problem of splice site prediction in DNA sequences. Experiments on two data sets with ten-fold cross validation demonstrated that method outperforms the benchmark methods: Naive Bayes classifier, SVM classifiers with linear kernel and polynomial kernel (dZ2 and dZ3) in terms of accuracy, precision, recall, and F-measure. The results confirmed that the proposed SVM-B method enhances the solution quality of Naive Bayes classifier for DNA splice site prediction. Furthermore, when the speed of computation was taken into consideration, the method was as quick as the Naive Bayes classifier and performed much faster than SVM with non-linear kernel methods. Solution quality, computational speed, user-friendliness, flexibility, and simplicity are some of the keys but conflicting factors in selecting and implementing new technology. The common industry practice is to trade off simplicity and speed of performance for solution quality. Study showed that by carefully selecting the proper encoding method, the linear SVMs can perform as well as complicated SVMs with polynomial kernels for splice site prediction, while maintaining computational efficiency. Therefore, with the proposed method there is no need to trade reduced accuracy for improved efficiency in SVMs applications (Zhang et al., 2005) 

 

In paper a new method was presented. It can identify human acceptor and donor splice sites. Combining orthogonal encoding with the sequential information and codon usage, the new method captures the sequence information and helps to achieve better results for human splice site prediction. Future work includes studying other features of splice sites and developing an effective classification algorithm to improve the accuracy of prediction (Wei et al., 2012). 

### 1.3 Machine learning methods  

Most of the existing computational methods encounter issues like lack of interpretability, improper separation of feature extraction and model training, manually defined features. These models rely majorly on handcrafted feature set construction. This manual operation for feature representation can be extremely laborious and expensive. The overhead of computational cost limits the dataset size for training the model, which has a chance of model underfitting. Moreover, concatenating features sets of different numerical can results in one high dimensional feature space, thereby difficult to apply machine learning methods. Deep learning, as the modern, leadingedge machine learning method has received ardent audience in the field of artificialintelligence. It has the ability to automatically extracts features from sequences and discovers complex representations of data patterns. Convolutional neural network (CNN) is one of the typical deep learning methods regarded as a cornerstone in the stream of deep learning models. It has achieved high classification performance in many areas like image processing, natural language processing, speech recognition and also in bioinformatics and computational biology problems. 

A large, deep convolutional neural network was trained to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0%, respectively, which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully connected layers with a final 1000-way softmax. To make training faster, we used nonsaturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully connected layers a recently developed regularization method called “dropout” employed that proved to be very effective. Also, a variant of this model in the ILSVRC-2012 was entered competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry (Krizhevsky et all., 2017). 

 

A rich set of deep, dynamic speech models is analyzed and summarized into two major categories: (1) top-down, generative models adopting localist representations of speech classes and features in the hidden space; and (2) bottom-up, discriminative models adopting distributed representations. With detailed examinations of and comparisons between these two types of models, there is focus on the localist versus distributed representations as their respective hallmarks and defining characteristics. Future directions are discussed and analysed about potential strategies to leverage the strengths of both the localist and distributed representations while overcoming their respective weaknesses, beyond blind integration of the two by using the generative model to pre-train the discriminative one as a popular method of training deep neural networks (Deng et al., 2014). 

Work shows that sequence specificities can be ascertained from experimental data with ‘deep learning’ techniques, which offer a scalable, flexible and unified computational approach for pattern discovery. Using a diverse array of experimental data and evaluation metrics, it is found that deep learning outperforms other state-of-the-art methods, even when training on in vitro data and testing on in vivo data. This approach is called DeepBind and have built a stand-alone software tool that is fully automatic and handles millions of sequences per experiment. Specificities determined by DeepBind are readily visualized as a weighted ensemble of position weight matrices or as a ‘mutation map’ that indicates how variations affect binding within a specific sequence (Alipanahi et al., 2015). 

CNN architecture was trained on promoters of five distant organisms: human, mouse, plant (Arabidopsis), and two bacteria (Escherichia coli and Bacillus subtilis). It was found that CNN trained on sigma70 subclass of Escherichia coli promoter gives an excellent classification of promoters and non-promoter sequences (Sn = 0.90, Sp = 0.96, CC = 0.84). The Bacillus subtilis promoters identification CNN model achieves Sn = 0.91, Sp = 0.95, and CC = 0.86. For human, mouse and Arabidopsis promoters we employed CNNs for identification of two well-known promoter classes (TATA and non-TATA promoters). CNN models nicely recognize these complex functional regions. For human promoters Sn/Sp/CC accuracy of prediction reached 0.95/0.98/0,90 on TATA and 0.90/0.98/0.89 for non-TATA promoter sequences, respectively. For Arabidopsis it was observed Sn/Sp/CC 0.95/0.97/0.91 (TATA) and 0.94/0.94/0.86 (non-TATA) promoters. Thus, the developed CNN models, implemented in CNNProm program, demonstrated the ability of deep learning approach to grasp complex promoter sequence characteristics and achieve significantly higher accuracy compared to the previously developed promoter prediction programs. It was also propose random substitution procedure to discover positionally conserved promoter functional elements. As the suggested approach does not require knowledge of any specific promoter features, it can be easily extended to identify promoters and other complex functional regions in sequences of many other and especially newly sequenced genomes (Umarov et al., 2017). 

In study “Human Splice-Site Prediction with Deep Neural Networks”, a new method of splice-site prediction using DNNs was proposed. The proposed system receives an input sequence data and returns an answer as to whether it is splice site. The length of input is 140 nucleotides, with the consensus sequence (i.e., ‘‘GT’’ and ‘‘AG’’ for the donor and acceptor sites, respectively) in the middle. Each input sequence model is applied to the pretrained DNN model that determines the probability that an input is a splice site. The model consists of convolutional layers and bidirectional long short-term memory network layers. The pretraining and validation were conducted using the data set tested in previously reported methods. The performance evaluation results showed that the proposed method can outperform the previous methods. In addition, the pattern learned by the DNNs was visualized as position frequency matrices (PFMs). Some of PFMs were very similar to the consensus sequence. The trained DNN model and the brief source code for the prediction system are uploaded. Further improvement will be achieved following the further development of DNNs (Naito et al., 2018). 


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
15. Sonnenburg S, R¨atsch G, Jagota A, M¨uller KR, New methods for splice site recognition, International Conference on Artificial Neural Networks, Springer, pp. 329–336, 2002
    
16. Alipanahi B, Delong A, Weirauch MT, Frey BJ, Predicting the sequence specificities of dna-and rna-binding proteins by deep learning, Nature biotechnology 33(8):831–838, 2015.
17. Bailey TL, Boden M, Buske FA, Frith M, Grant CE, Clementi L, Ren J, Li WW,
Noble WS, Meme suite: tools for motif discovery and searching, Nucleic acids research
37(suppl 2):W202–W208, 2009.
18. Baten AK, Chang BC, Halgamuge SK, Li J, Splice site identification using probabilistic parameters and svm classification, BMC bioinformatics, Springer, p. S15, 2006.
19. Baten AK, Halgamuge SK, Chang BC, Fast splice site detection using information
content and feature reduction, BMC bioinformatics 9(S12):S8, 2008.
20. Benson DA, Cavanaugh M, Clark K, Karsch-Mizrachi I, Lipman DJ, Ostell J, Sayers
EW, Genbank, Nucleic acids research 41(D1):D36–D42, 2012.
21. Burge C, Karlin S, Prediction of complete gene structures in human genomic dna,
Journal of molecular biology 268(1):78–94, 1997.
22. Chen TM, Lu CC, Li WH, Prediction of splice sites with dependency graphs and their
expanded bayesian networks, Bioinformatics 21(4):471–482, 2005.
23. Chollet F, Xception: Deep learning with depthwise separable convolutions, Proceedings
of the IEEE conference on computer vision and pattern recognition, pp. 1251–1258, 2017.
24. Cloonan N, Forrest AR, Kolle G, Gardiner BB, Faulkner GJ, Brown MK, Taylor DF,
Steptoe AL, Wani S, Bethel G, et al., Stem cell transcriptome profiling via massivescale mrna sequencing, Nature methods 5(7):613, 2008.
10. Davis J, Goadrich M, The relationship between precision-recall and roc curves, Proceedings of the 23rd international conference on Machine learning, pp. 233–240, 2006.
25. Degroeve S, Saeys Y, De Baets B, Rouz´e P, Van De Peer Y, Splicemachine: predicting splice sites from high-dimensional local context representations, Bioinformatics 21(8):1332–1338, 2005.
26. Deng L, Togneri R, Deep dynamic models for learning hidden representations of speech
features, in Speech and audio processing for coding, enhancement and recognition, ,
Springer, pp. 153–195, 2015.
27. Du X, Yao Y, Diao Y, Zhu H, Zhang Y, Li S, Deepss: Exploring splice site motif through convolutional neural network directly from dna sequence, IEEE Access 6:32958–32978, 2018.
28. Gupta S, Stamatoyannopoulos JA, Bailey TL, Noble WS, Quantifying similarity between motifs, Genome biology 8(2):R24, 2007.
29. Ho LS, Rajapakse JC, Splice site detection with a higher-order markov model implemented on a neural network, Genome Informatics 14:64–72, 2003.
30. Kamath U, De Jong K, Shehu A, Effective automated feature construction and selection for classification of biological sequences, PloS one 9(7), 2014
    
31. Kim D, Langmead B, Salzberg SL, Hisat: a fast spliced aligner with low memory
requirements, Nature methods 12(4):357–360, 2015.
32. Kiranyaz S, Avci O, Abdeljaber O, Ince T, Gabbouj M, Inman DJ, 1d convolutional
neural networks and applications: A survey, arXiv preprint arXiv:190503554 , 2019.
33. Krizhevsky A, Sutskever I, Hinton GE, Imagenet classification with deep convolutional
neural networks, Advances in neural information processing systems, pp. 1097–1105, 2012.
34. Lee B, Lee T, Na B, Yoon S, Dna-level splice junction prediction using deep recurrent
neural networks, arXiv preprint arXiv:151205135 , 2015.
35. Maji S, Garg D, Hybrid approach using svm and mm2 in splice site junction identification, Current Bioinformatics 9(1):76–85, 2014.
36. Meher PK, Sahu TK, Rao A, Wahi S, Identification of donor splice sites using support
vector machine: a computational approach based on positional, compositional and
dependency features, Algorithms for molecular biology 11(1):16, 2016.
37. Meher PK, Sahu TK, Rao AR, Prediction of donor splice sites using random forest
with a new sequence encoding approach, BioData mining 9(1):4, 2016.
38. Meher PK, Sahu TK, Rao AR, Wahi SD, A statistical approach for 5 splice site
prediction using short sequence motifs and without encoding sequence data, BMC
bioinformatics 15(1):362, 2014.
39. Mortazavi A, Williams BA, McCue K, Schaeffer L, Wold B, Mapping and quantifying
mammalian transcriptomes by rna-seq, Nature methods 5(7):621, 2008.
40. Naito T, Human splice-site prediction with deep neural networks, Journal of Computational Biology 25(8):954–961, 2018.
41. Pertea M, Lin X, Salzberg SL, Genesplicer: a new computational method for splice
site prediction, Nucleic acids research 29(5):1185–1190, 2001.
42. Pollastro P, Rampone S, Hs3d, a dataset of homo sapiens splice regions, and its
extraction procedure from a major public database, International Journal of Modern Physics C 13(08):1105–1117, 2002.
43. Rajapakse JC, Ho LS, Markov encoding for detecting signals in genomic sequences,
IEEE/ACM Transactions on Computational Biology and Bioinformatics 2(2):131–142, 2005.
44. Sandelin A, Alkema W, Engstr¨om P, Wasserman WW, Lenhard B, Jaspar: an openaccess database for eukaryotic transcription factor binding profiles, Nucleic acids research 32(suppl 1):D91–D94, 2004.
45. Sheth N, Roca X, Hastings ML, Roeder T, Krainer AR, Sachidanandam R, Comprehensive splice-site analysis using comparative genomics, Nucleic acids research 34(14):3955–3967, 2006.
46. Sonnenburg S, R¨atsch G, Jagota A, M¨uller KR, New methods for splice site recognition, International Conference on Artificial Neural Networks, Springer, pp. 329–336, 2002.
47. Sonnenburg S, Schweikert G, Philips P, Behr J, R¨atsch G, Accurate splice site prediction using support vector machines, BMC bioinformatics, Springer, p. S7, 2007.
48. Staden R, Computer methods to locate signals in nucleic acid sequences , 1984.
49. Sutskever I, Vinyals O, Le QV, Sequence to sequence learning with neural networks,
Advances in neural information processing systems, pp. 3104–3112, 2014.
50. Umarov RK, Solovyev VV, Recognition of prokaryotic and eukaryotic promoters using
convolutional deep learning neural networks, PloS one 12(2), 2017.
51. Wang K, Singh D, Zeng Z, Coleman SJ, Huang Y, Savich GL, He X, Mieczkowski P, Grimm SA, Perou CM, et al., Mapsplice: accurate mapping of rna-seq reads for splice junction discovery, Nucleic acids research 38(18):e178–e178, 2010

    
52. Wang R, Wang Z, Wang J, Li S, Splicefinder: ab initio prediction of splice sites using
convolutional neural network, BMC bioinformatics 20(23):652, 2019.
53. Wei D, Zhang H, Wei Y, Jiang Q, A novel splice site prediction method using support vector machine, Journal of Computational Information Systems 9(20):8053–8060, 2013.
54. Wei D, Zhuang W, Jiang Q, Wei Y, A new classification method for human gene splice
site prediction, International Conference on Health Information Science, Springer, pp. 121–130, 2012.
55. Wu TD, Nacu S, Fast and snp-tolerant detection of complex variants and splicing in
short reads, Bioinformatics 26(7):873–881, 2010.
56. Yeo G, Burge CB, Maximum entropy modeling of short sequence motifs with applications to rna splicing signals, Journal of computational biology 11(2-3):377–394, 2004.
57. Zhang Y, Chu CH, Chen Y, Zha H, Ji X, Splice site prediction using support vector
machines with a bayes kernel, Expert Systems with Applications 30(1):73–81, 2006.
58. Zhang Y, Liu X, MacLeod JN, Liu J, Deepsplice: Deep classification of novel splice
junctions revealed by rna-seq, 2016 IEEE international conference on bioinformatics
and biomedicine (BIBM), IEEE, pp. 330–333, 2016
