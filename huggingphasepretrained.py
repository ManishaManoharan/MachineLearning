from transformers import pipeline

# Load the pretrained spam model
classifier = pipeline(
    "text-classification",
    model="SGHOSH1999/bert-email-spam-classifier_tuned"
)

# Mapping from LABEL_0/LABEL_1 to Spam/Ham
label_mapping = {
    "LABEL_0": "Not Spam",
    "LABEL_1": "Spam"
}

while True:
    user_input = input("\nEnter an email/message (type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Program ended.")
        break

    result = classifier(user_input)[0]
    label = result['label']
    score = result['score']

    mapped_label = label_mapping.get(label, "Other")
    print(f"Prediction: {mapped_label} (Confidence: {score:.2f})")
