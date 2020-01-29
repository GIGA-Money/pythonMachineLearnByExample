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

from nltk.tokenize import word_tokenize
import spacy


sent = '''reading a book, python'''

'''
word_tokenize function keeps punctuation marks and digits, 
only discards whitespaces and newlines.
'''
print(word_tokenize(sent))

sent2 = '''Pasted on the wall is a sharp looking paint spot HELP!!!'''
print(word_tokenize(sent2))

# End of ex:1.

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
print([token.text for token in tokens2])

