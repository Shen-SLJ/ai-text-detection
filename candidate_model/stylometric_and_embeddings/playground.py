from nltk.tokenize import sent_tokenize

text = "Dr. Smith went to the U.S.A. for his appointment. He plans to return on Aug. 28th."

sentences = sent_tokenize(text)

print(sentences)