# Matchmaker Input Data Formats

In general every data input file in matchmaker is a .tsv file with tabs separating columns, as found in MSMARCO v1. We choose this format, because it allows us to easily open and operate on files, and replacing tabs in the text does not alter the meaning of it. If you have a collection in another format such as older TREC collections or CORD-19 have a look in the *preprocessing/convert_formats* folder for conversion scripts. 

In the following we describe the data format expected for certain operations of matchmaker:

**It is important to have the exact number of columns in every line that is expected, otherwise our data loaders will throw an exception (to not make assumptions that come back to haunt us) -> so clear tabs and newlines from text before converting it into the .tsv format**

## Training

### Static Sampling (Default)

For static sampling (e.g. already sampled query, rel. passage, non-rel. passage triples) we set the config ``train_tsv: "/path/to/train/triples.tsv"``

The format of the .tsv file depends on the training type: standalone or with pairwise distillation (which is activated with the config: ``train_pairwise_distillation: True``)

**Standalone format (per line):**

````
query-text<tab>pos-text<tab>neg-text
````

**Distillation format  (per line):**

````
score-pos<tab>score-neg<tab>query-text<tab>pos-text<tab>neg-text
````

### Dynamic Sampling - TAS-Balanced

For dynamic sampling, currently implemented TAS-Balanced (which is activated with the config: ``dynamic_sampler: True``) we need to set all data files individually:

````
dynamic_query_file: "/path/to/train/queries.train.tsv"
dynamic_collection_file: "/path/to/train/collection.tsv"
dynamic_pairs_with_teacher_scores: "/path/to/train/T2-train-ids.tsv"
dynamic_query_cluster_file: "/path/to/train/train_clusters.tsv"
````

The dynamic_collection_file file:

````
doc-id<tab>doc-text
````

The dynamic_query_file file:

````
query-id<tab>query-text
````

The dynamic_pairs_with_teacher_scores file:

````
scorpos<tab>scoreneg<tab>query<tab>docpos<tab>docneg
````

The dynamic_query_cluster_file file, contains all query ids for a single cluster per line (there is no fixed size in terms of number of queries or number of clusters)

````
c1-query-id1 c1-query-id2 c1-query-id3 c1-query-id4 ....
c2-query-id1 c2-query-id2 ..
...
````


## Re-Ranking Evaluation

For re-ranking (including the validation set) we need candidate files from a first stage retriever (fe. BM25 with Pyserini), with those candidate files we can create the input file in the following format:

````
query-id<tab>doc-id<tab>query-text<tab>doc-text
````

## Dense Retrieval Evaluation

The collection file:

````
doc-id<tab>doc-text
````

The query file:

````
query-id<tab>query-text
````