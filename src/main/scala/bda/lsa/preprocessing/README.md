# LDA executions

We executed our whole pipeline multiple times.
This file presents what we did and the results we got :

## Execution with 1520 documents
We arbitrarily selected a total of 1520 Wikipedia documents.

Here are the different steps we executed :

### Input
We downloaded an XML version of 1520 pages on the [Special:Export](https://en.wikipedia.org/wiki/Special:Export) page of Wikipedia.
It lets us choose exactly what pages we want to extract.
We built a subset using the pages contained in the following categories :

- 

### Execution 1

#### Tokenizing
We chose to use `org.apache.spark.ml.feature.Tokenizer` to tokenize our input.  
We got a total of **2'712'636 tokens**, of which 201'563 are unique.  
This makes for an average of **1'784 tokens per document**, of which 694 are unique.

#### Removing stop words
We then removed all the stopwords we found from that set.  
We got a total of **1'722'062 tokens**, of which 201'427 are unique.  
That means we removed the occurences of 156 stopwords.  
This makes for an average of **1'133 tokens per document**. of which 638 are unique.

### Execution 2

#### Lemmatizing
We tokenized using Standford's NLP to tokenize, and then we lemmatized the results.  
We got a total of **2'021'543 tokens**, of which 109'995 are unique.
This makes for an average of **1'399 tokens per document**. of which 518 are unique.

#### Filtering
We only kept words that are composed of only letters, and of length at least two.
We got a total of **1'458'090 tokens**, of which 70'638 are unique.  
This makes for an average of **960 tokens per document**, of which 439 are unique.

Here is a table summarizing these informations :

|                    	|    Tokens 	|  Unique 	| Tokens/document 	| Unique/document 	|
|--------------------	|----------:	|--------:	|----------------:	|----------------:	|
| Tokenizing         	| 2'712'636 	| 201'563 	|           1'784 	|             694 	|
| Removing Stopwords 	| 1'722'062 	| 201'427 	|           1'133 	|             638 	|
| Lemmatizing        	| 2'021'543 	| 109'995 	|           1'399 	|             518 	|
| Filtering          	| 1'458'090 	|  70'638 	|             960 	|             439 	|

### Other stats

We have 1'519 / 1'520 documents that contain at least 1 term.