from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Student feedback
feedback = [
    "The course was very interesting and useful.",
    "The teaching was excellent.",
    "I did not like the course.",
    "The lectures were difficult to understand.",
    "The faculty explained the topics clearly."
]

# Tokenize each feedback
for sentence in feedback:
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    print("Sentence:", sentence)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    print("-" * 60)