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

### Tokenizing
We chose to use `org.apache.spark.ml.feature.Tokenizer` to tokenize our input.  
We got a total of **2'712'636 tokens**, of which 201'563 are unique.  
This makes for an average of **1'784 tokens per document**, of which 694 are unique.

### Removing stop words
We then removed all the stopwords we found from that set.  
We got a total of **1'722'062 tokens**, of which 201'427 are unique.  
That means we removed the occurences of 156 stopwords.  
This makes for an average of **1'133 tokens per document**. of which 638 are unique.

### Lemmatizing
After this, we lemmatized the results.  
We got a total of **X tokens**, of which X are unique.
This makes for an average of **X tokens per document**. of which XS are unique.

### Filtering
We only kept words that are composed of only letters, and of length at least two.
We got a total of **1'458'090 tokens**, of which 70'638 are unique.  
This makes for an average of **960 tokens per document**, of which 439 are unique.