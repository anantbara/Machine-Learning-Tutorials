
# Natural Language Processing

Natural Language Processing is a way for computers to analyze, understand, and derive meaning from human language in a smart and useful way. Processing of Natural Language is required when you want an intelligent system like robot to perform as per your instructions, when you want to hear decision from a dialogue based clinical expert system, etc. With NLP, you can solve many problems like automatic summarization, translation, named entity recognition, relationship extraction, sentiment analysis, speech recognition, and topic segmentation. Few of these tasks we will see in next tutorial. For now, I want to discuss few basic tasks which are very helpful to process the text.

## Tokenization

Your corpus(Training data) may have a single line of text or it contains multiple lines. There are two operations in Tokenization i.e. Word tokenization and Sentence Tokenization. Let's see below how we can perform this operation in python. To do so, you need Python and NLTK package installed on your machine. 


```python
# Tokenize Text into words
from nltk.tokenize import word_tokenize

text = "Students of this college are really awesome."
word_tokens = word_tokenize(text)
print("No. of words : ",len(word_tokens))
print(word_tokens)
```

    No. of words :  8
    ['Students', 'of', 'this', 'college', 'are', 'really', 'awesome', '.']
    


```python
# Tokenize Text into sentences
from nltk.tokenize import sent_tokenize

text = "Students of this college are really awesome. Because they are very serious about their classes, they always get good rank in their university."
sent_tokens = sent_tokenize(text)
print("No. of Sentences : ",len(sent_tokens))
print(sent_tokens)
```

    No. of Sentences :  2
    ['Students of this college are really awesome.', 'Because they are very serious about their classes, they always get good rank in their university.']
    

## POS Tagging

Like in English language, we have tags like Noun, Verb, Adjective and so on, we have POS tag to make machine understand the text which is standardized by the University of Pennsylvania (also known as Penn Treebank). To know more about these tags [click here](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). Let me show you through the code give below.


```python
import nltk

tags = nltk.pos_tag(word_tokens)
print(tags)
```

    [('Students', 'NNS'), ('of', 'IN'), ('this', 'DT'), ('college', 'NN'), ('are', 'VBP'), ('really', 'RB'), ('awesome', 'JJ'), ('.', '.')]
    

## Chunking

In chunking, we have to write rules for grouping these tagged words. This will help you to extract the properties of the subject. For example, Cat is a noun but "Black cat" is not a proper noun. "Black" is property (color) of the noun word (Cat). So if you want to consider these two words as a noun phrase then you have to write a rule to extract such information.


```python
sent = "The prime minister of india is doing well for his country"
word_tokens = nltk.word_tokenize(sent)
taged_words = nltk.pos_tag(word_tokens)

grammar = "NP: {<DT>?<JJ>*<NN>?<IN>?<NN>}"
chunker = nltk.RegexpParser(grammar)
chunked_sent = chunker.parse(taged_words)
chunked_sent.draw()
```

## Parsing

Here you can write your own grammer rule which is used to parse a sentence and extract enteties out of it.


```python
grammar1 = nltk.CFG.fromstring("""
  S -> NP VP
  VP -> V NP | V NP PP | V N
  PP -> P NP
  V -> "saw" | "ate" | "walked"
  NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
  Det -> "a" | "an" | "the" | "my"
  N -> "man" | "dog" | "apple" | "telescope" | "park"
  P -> "in" | "on" | "by" | "with"
  """)

sent = "John ate apple".split()
rd_parser = nltk.RecursiveDescentParser(grammar1)
for tree in rd_parser.parse(sent):
    print(tree)
```

    (S (NP John) (VP (V ate) (N apple)))
    
