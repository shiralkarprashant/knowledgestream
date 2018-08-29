# Paper
Code to reproduce results in the paper "Finding Streams in Knowledge Graphs to Support Fact Checking" in proceedings of ICDM 2017. A full version of this paper can be found at: https://arxiv.org/abs/1708.07239

# Fetch
```
git clone https://github.com/shiralkarprashant/knowledgestream
cd knowledgestream
```

# Data
Download data from the following URL http://carl.cs.indiana.edu/data/fact-checking/data.zip and decompress it inside `knowledgestream` directory. This compressed file contains three items: 

1. DBpedia 2016-10 knowledge graph represented by three files: nodes.txt, relations.txt, and abbrev_cup_dbpedia_filtered.nt, which have been derived from raw data on the DBpedia downloads page: [DBpedia 2016-10](http://wiki.dbpedia.org/downloads-2016-10). Additionally, it contains a directory named `_undir` which contains datastructures in binary format as required by the code. If you are interested in applying methods in this repository on your own knowledge graph, you may use the following script to generate the required graph files (.npy): [KG generation script](https://github.com/shiralkarprashant/knowledgestream/blob/master/datastructures/test_graph.py)
2. A collection of synthetic and real datasets, some of which were created by us, while others were downloaded from [KGMiner](https://github.com/nddsg/KGMiner/) or as provided by Google and the WSDM Cup 2017 Triple Scoring challenge organizers. The true triples (positive examples) in the synthetic datasets created based on Wikipedia lists given below, and the false triples (negative examples) were created by "perturbing" the set of objects in each list; this is also called "local-closed world assumption (LCWA)."
	- NBA-Team: [List of NBA players who have spent their entire career with one franchise](https://en.wikipedia.org/wiki/List_of_NBA_players_who_have_spent_their_entire_career_with_one_franchise)
	- Oscars: [Academy Award for Best Picture](https://en.wikipedia.org/wiki/Academy_Award_for_Best_Picture)
	- FLOTUS: [List of First Ladies of the United States](https://en.wikipedia.org/wiki/List_of_First_Ladies_of_the_United_States)
	- World Capitals: [List of national capitals in alphabetical order](https://en.wikipedia.org/wiki/List_of_national_capitals_in_alphabetical_order)
	- Birthplace-Deathplace: This was created just based on DBpedia. Persons having different birth and death place were identified and 250 individuals were sampled from five buckets partitioning [Birthplace-Deathplace](http://carl.cs.indiana.edu/data/fact-checking/histogram_persons_vs_facts.pdf) distribution. Their death place was forged as a false example (or triple) of their birth place, while their birth place was taken as a true triple, thereby creating 250 true and 250 false triples. 
3. A relational similarity matrix obtained using TF-IDF representation of relations in the knowledge graph. See paper for details.

# Install
```python setup.py build_ext -if```

```python setup.py install```

Note: for the second command, please do sudo in case you need installation rights on the machine.


# Run 

## Knowledge Stream (KS)

```kstream -m stream -d datasets/sample.csv -o output/```

You should see output such as 

```
[19:00:10] Launching stream..
[19:00:10] Dataset: sample.csv
[19:00:10] Output dir: output
[19:00:10] Read data: (5, 7) sample.csv
[19:00:10] Note: Found non-NA records: (5, 7)
Reconstructing graph from data/kg/_undir
=> Loaded: undir_data.npy
=> Loaded: undir_indptr.npy
=> Loaded: undir_indices.npy
=> Loaded: undir_indeg_vec.npy
=> Graph loaded: 11.63 secs.

[19:00:22] Computing KNOWLEDGE-STREAM for 5 triples..
1. Working on (2985653, 599, 3218267) .. [19:00:38] mincostflow: 0.08863, #paths: 5, time: 12.55s.
 2. Working on (2734002, 599, 5305646) .. [19:00:52] mincostflow: 0.15648, #paths: 5, time: 12.91s.
 3. Working on (5140024, 599, 4567127) .. [19:01:01] mincostflow: 0.16628, #paths: 5, time: 9.20s.
 4. Working on (1522148, 599, 1357357) .. [19:01:11] mincostflow: 0.09414, #paths: 5, time: 9.28s.
 5. Working on (4319468, 599, 2450828) .. [19:01:19] mincostflow: 0.16025, #paths: 5, time: 8.05s.
[19:01:19] * Saved results: output/out_kstream_sample_2017-08-23.csv
[19:01:19] Mincostflow computation complete. Time taken: 57.64 secs.
```
and a CSV file is created at the specified output directory, which contains `score` and `softmaxscore` (normalized) for each triple.

## Relational Knowledge Linker (KL-REL)

```kstream -m relklinker -d datasets/sample.csv -o output/```

You should see output such as 

```
[18:56:43] Launching relklinker..
[18:56:43] Dataset: sample.csv
[18:56:43] Output dir: output
[18:56:43] Read data: (5, 7) sample.csv
[18:56:43] Note: Found non-NA records: (5, 7)
Reconstructing graph from data/kg/_undir
=> Loaded: undir_data.npy
=> Loaded: undir_indptr.npy
=> Loaded: undir_indices.npy
=> Loaded: undir_indeg_vec.npy
=> Graph loaded: 1.24 secs.

[18:56:44] Computing REL-KLINKER for 5 triples..
[18:56:50] time: 3.35s
1. Working on (2985653, 599, 3218267)..[18:56:55] time: 2.90s
 2. Working on (2734002, 599, 5305646)..[18:56:57] time: 2.18s
 3. Working on (5140024, 599, 4567127)..[18:57:00] time: 2.16s
 4. Working on (1522148, 599, 1357357)..[18:57:02] time: 2.15s
 5. Working on (4319468, 599, 2450828)..[18:57:03] 
[18:57:03] * Saved results: output/out_relklinker_sample_2017-08-23.csv
[18:57:03] Relklinker computation complete. Time taken: 18.68 secs.
```
and a CSV file is created at the specified output directory, which contains `score` and `softmaxscore` (normalized) for each triple.

## Knowledge Linker (KL)

```kstream -m relklinker -d datasets/sample.csv -o output/```

You should see output such as 


```
[16:02:18] Launching klinker..
[16:02:18] Dataset: sample.csv
[16:02:18] Output dir: output
[16:02:18] Read data: (3, 7) sample.csv
[16:02:18] Note: Found non-NA records: (3, 7)
Reconstructing graph from data/kg/_undir
=> Loaded: undir_data.npy
=> Loaded: undir_indptr.npy
=> Loaded: undir_indices.npy
=> Loaded: undir_indeg_vec.npy
=> Graph loaded: 0.54 secs.

[16:02:19] Computing KL for 3 triples..
1. Working on (392035, 599, 2115741).. time: 3.09s
2. Working on (392035, 599, 4119746).. time: 3.82s
3. Working on (392035, 599, 917821).. time: 3.13s
...
[16:02:34] * Saved results: output/out_klinker_sample_2017-08-25.csv
[16:02:34] KL computation complete. Time taken: 15.29 secs.
```

and a CSV file is created at the specified output directory, which contains `score` and `softmaxscore` (normalized) for each triple.

Alternatively, code for this method can be found at [Knowledge Linker](https://github.com/glciampaglia/knowledge_linker). 

## Predicate Path Mining (PredPath)

```kstream -m 'predpath' -d ./datasets/sample.csv -o ./output/```

You should see output such as 

```
[13:20:16] Launching predpath..
[13:20:16] Dataset: sample.csv
[13:20:16] Output dir: output
[13:20:16] Read data: (104, 7) sample.csv
[13:20:16] Note: Found non-NA records: (104, 7)
Reconstructing graph from data/kg/_undir
=> Loaded: undir_data.npy
=> Loaded: undir_indptr.npy
=> Loaded: undir_indices.npy
=> Loaded: undir_indeg_vec.npy
=> Graph loaded: 3.82 secs.

=> Removing predicate 599.0 from KG.

=> Path extraction..(this can take a while)

P: +:40, -:33, unique tot:73
Time taken: 160.79s

=> Path selection..
D: +:40, -:33, tot:73
Time taken: 0.07s

=> Model building..
#Features: 73, best-AUROC: 0.92943
Time taken: 0.65s

Time taken: 163.72s

Saved: output/out_predpath_sample_2017-08-25.pkl
```

# Path Ranking Algorithm (PRA)

```kstream -m 'pra' -d ./datasets/sample.csv -o ./output/```

You should see output such as 

```
[13:31:37] Launching pra..
[13:31:37] Dataset: sample.csv
[13:31:37] Output dir: output
[13:31:37] Read data: (104, 7) sample.csv
[13:31:37] Note: Found non-NA records: (104, 7)
Reconstructing graph from data/kg/_undir
=> Loaded: undir_data.npy
=> Loaded: undir_indptr.npy
=> Loaded: undir_indices.npy
=> Loaded: undir_indeg_vec.npy
=> Graph loaded: 1.87 secs.

=> Removing predicate 599 from KG.

=> Path extraction..
...
#Features: 73
Time taken: 75.98s

=> Path selection..
Selected 73 features

=> Constructing feature matrix..
Time taken at C level: 2.656s
Time taken: 2.87590s

=> Model building..
#Features: 73, best-AUROC: 0.7786
Time taken: 0.59s

Time taken: 81.39s

Saved: output/out_pra_sample_2017-08-25.pkl
```

# TransE

Code for this method and couple other methods based on the idea of knowledge graph embedding can be found at [KB2E](https://github.com/thunlp/KB2E).

# Link prediction algorithms: Katz, PathEnt, SimRank, Adamic & Adar, Jaccard, Degree Product

All link prediction algorithms can be invoked in a similar manner. The method names to specify are respectively: ``katz``, ``pathent``, ``simrank``, ``adamic_adar``, ``jaccard``, and ``degree_product``. 

Example: ```kstream -m 'katz' -d ./datasets/sample.csv -o ./output/```

You should see output such as 

```
[15:25:06] Launching katz..
[15:25:06] Dataset: sample.csv
[15:25:06] Output dir: output
[15:25:06] Read data: (9, 7) sample.csv
[15:25:06] Note: Found non-NA records: (9, 7)
Reconstructing graph from data/kg/_undir
=> Loaded: undir_data.npy
=> Loaded: undir_indptr.npy
=> Loaded: undir_indices.npy
=> Loaded: undir_indeg_vec.npy
=> Graph loaded: 8.79 secs.

[15:25:16] Computing KZ for 9 triples..
1. Working on (392035, 599, 2115741).. score: 0.12750, time: 3.74s
2. Working on (392035, 599, 4119746).. score: 0.02813, time: 3.99s
3. Working on (392035, 599, 917821).. score: 0.03825, time: 3.59s
...
* Saved results: output/out_katz_sample_2017-08-25.csv
```

and a CSV file is created at the specified output directory, which contains `score` and `softmaxscore` (normalized) for each triple.
