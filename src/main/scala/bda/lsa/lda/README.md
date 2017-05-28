# Running LDA

LDA stands for *Latent Dirichelet Allocation*.
It is a generative probabilistic model that is used in natural language processing.

## Our usage
In our case we'll use it to retrieve concepts from Wikipedia articles.
That means we'll be using the model not by generating documents, but rather by analysing the collection we have and try to infer what are the common topics of a set of them.
We'll then try to show interesting statistics about the topics we found.

## LDA Parameters

LDA uses some values that are customizable, and should fit to the chosen situation.
Here they are :

- K, which is `k` in our code. Represents the number of topics.
- V, which is ``


