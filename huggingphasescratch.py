# Email Spam Detection using Naive Bayes (From Scratch)

spam_texts = [
    "congratulations you have won a cash prize",
    "claim your free coupon now",
    "limited offer buy now and save big",
    "urgent your account has been compromised",
    "win a brand new phone by clicking this link"
]

ham_texts = [
    "meeting scheduled at 10 am tomorrow",
    "please find the attached assignment",
    "lets catch up over coffee this weekend",
    "project deadline has been extended",
    "reminder submit your lab record today"
]

def tokenize(text):
    return text.lower().split()

# Build word lists
spam_words = []
ham_words = []

for text in spam_texts:
    spam_words.extend(tokenize(text))

for text in ham_texts:
    ham_words.extend(tokenize(text))

# Vocabulary
vocab = set(spam_words + ham_words)

# Word frequency counts
spam_count = {}
ham_count = {}

for word in vocab:
    spam_count[word] = spam_words.count(word)
    ham_count[word] = ham_words.count(word)

# Prior probabilities
spam_prior = 0.5
ham_prior = 0.5

def predict(text):
    words = tokenize(text)

    # Check if any word exists in vocabulary
    known_words = [word for word in words if word in vocab]

    if len(known_words) == 0:
        return "Other (Unknown)"

    spam_prob = spam_prior
    ham_prob = ham_prior

    for word in known_words:
        spam_prob *= (spam_count[word] + 1) / (len(spam_words) + len(vocab))
        ham_prob *= (ham_count[word] + 1) / (len(ham_words) + len(vocab))

    if spam_prob > ham_prob:
        return "Spam"
    else:
        return "Not Spam"
    

    # User input loop
while True:
    user_input = input("\nEnter an email/message (type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("Program ended.")
        break

    print("Prediction:", predict(user_input))


