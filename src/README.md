# Deep Splicer

Deep Splicer is a deep learning model based on convolutional neural networks (CNN) for the identification of splice sites in the genomes of humans and other species.
It was constructed and evaluated using the human genome (*Homo sapiens*) and was additionally assessed using other species’ genomes to identify how well a model trained on the human genome can generalize to other species since the objective is to use this model as a first-line annotation tool for newly sequenced genomes. The genomes of other species that were used to evaluate Deep Splicer were from *Mus musculus*, *Danio rerio*, *Drosophila melanogaster*, *Arabidopsis thaliana* and *Caenorhabditis elegans*.

To run Deep Splicer, clone the repository using:

```
git clone https://github.com/ElisaFernandezCastillo/DeepSplicer.git
```

Run the code in `Deep_Splicer.ipynb` following the order of the execution cells (from top to bottom) using a platform like Jupyter Notebook or Google Colaboratory. 

Alternatively, the Keras model can be loaded using the `deep_splicer.h5` file located on the `models` directory following this example: 

```
from keras.models import load_model
model = load_model("deep_splicer.h5")
```

The Python functions located on **Predicting on individual genomic sequences** section on `Deep_Splicer.ipynb` can be used for doing predictions. For example, the function `testing_transcripts(transcript_id= ‘<Write Gene Ensembl ID>’ model= <Deep Splicer model> )` receives an Ensembl Gene Id as a string and the model as input parameters and does the predictions as shown below:

```
testing_transcripts(transcript_id='ENSMUSG00000029120', model=model)
Intron positions on string:  [(5378, 59454), (59553, 62771), (62938, 64017), (64131, 67722), (67901, 76583), (76749, 83554), (83725, 86039), (86132, 88837)]
Number of sequences predicted per index (donor, acceptor, other):  [  155   195 96216]
Intron position predicted by the model:  ({5378: 5, 59553: 2, 62938: 6, 64131: 24, 67901: 3, 76749: 0, 83725: 1, 86132: 10}, {'top-k': 0.75, 'top-10%': 0.88, 'top-25%': 1.0, 'top-50%': 1.0, 'top-65%': 1.0, 'top-75%': 1.0, 'top-85%': 1.0, 'top-95%': 1.0})
Intron position predicted by the model:  ({59454: 5, 62771: 3, 64017: 1, 67722: 0, 76583: 4, 83554: 6, 86039: 7, 88837: 2}, {'top-k': 1.0, 'top-10%': 1.0, 'top-25%': 1.0, 'top-50%': 1.0, 'top-65%': 1.0, 'top-75%': 1.0, 'top-85%': 1.0, 'top-95%': 1.0})
```
