# Use for retrieving data sets.
# import nltk
# nltk.download()

# Use for testing data set.
#  from nltk.corpus import names

# Use to test dataset is working.
# print(names.words()[:10])

# Use to print work length.
# print(len(names.words()))

# Beginning of ex:0.
# Use to test the implementation of work-based tokenizing using the work_tokenize function in NLTK.

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

sent = '''reading a book, python'''

'''
word_tokenize function keeps punctuation marks and digits, 
only discards whitespaces and newlines.
'''
print(word_tokenize(sent))

sent2 = '\nPasted on the wall is a sharp looking paint spot' \
        ' \nHELP!!!'
print("ex0", word_tokenize(sent2))

# End of ex:0.

# the use of the SpaCy lib also has an tokenizing feature.
# Uses an accurate trained model that is constantly updated.

# import spacy
'''
well put all the imports at the top
also be sure to 'python -m spacy download en_core_web_sm' 
in the pycharm terminal if not done yet
'''

nip = spacy.load('en_core_web_sm')
tokens2 = nip(sent2)
print('ex1', [token.text for token in tokens2])

'''
to segment text based on sentence,
from the NLTK lib we have sent_tokenize
'''
print('ex1', sent_tokenize(sent), sent_tokenize(sent2))

# Two sentence-based tokens are returned,
# as there are two sentences in the input text regardless of a newline following a comma.

# End of ex:1

# beginning of ex:2

# PoS tagging:
'''
We can apply an off-the-shelf tagger from NLTK or combine multiple taggers 
to customize the tagging process
'''
tokens = word_tokenize(sent)
print('ex2', nltk.pos_tag(tokens))

# The PoS tag following each token is returned.
# We can check the meaning of a tag using the help function.
# run the code to see results.
nltk.help.upenn_tagset('PRP')
nltk.help.upenn_tagset('VBP')
nltk.help.upenn_tagset('DT')
nltk.help.upenn_tagset('NN')

# In spaCy, getting a PoS tag is also easy. The token object parsed from an input sentence has an attribute called pos_,
# which is the tag we are looking for:
print('ex2', [(token.text, token.pos_) for token in tokens2])

# end of ex:2

# beginning of ex:3

# Named-entity recognition
'''
Given a text sequence, the named-entity recognition (NER) task is to locate and identify words or phrases that are of 
definitive categories such as names of persons, companies, locations, and dates.
'''

# Using spaCy for NER or named-entity recognition.

# well tokenize the a new sentence to have a wide character variety and symbol variety.

sent3 = 'As produced in 1872, buffalo bill or william cody,' \
        'would sell tickets to his wild west show "The Scouts of the Prairie" ' \
        'after all of his shows ended there were around 2.5 million tickets sold'
tokens3 = nip(sent3)
print('ex3', [(token_ent.text, token_ent.label_) for token_ent in tokens3.ents])
'''
when run, we see that the date 1872 is marked as DATE,
the show name is marked as a WORK_OF_ART,
ad the count of tickets sold is CARDINAL. 
(as cardinal numbers are a form of measurement, or set indicator)
'''

# end of ex:3

# beginning of ex:4

# Stemming and Lemmatization
'''
Word stemming is a process of reverting an inflected or derived word to its root form.
The word lemmatization is a cautious version of stemming.
'''
# Page 70.