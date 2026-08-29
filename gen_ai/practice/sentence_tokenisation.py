from nltk.tokenize import word_tokenize

paragraph = input("Enter a paragraph: ")

tokens = word_tokenize(paragraph)

print("Tokens:")
for token in tokens:
    print(token)
