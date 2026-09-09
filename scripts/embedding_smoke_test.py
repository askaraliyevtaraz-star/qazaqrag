from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"


model = SentenceTransformer(MODEL_NAME)


texts = [
    "passage: The university library is open until 10 PM.",
    "passage: Сегодня на улице идёт дождь.",
    "passage: Студенттік клубқа бес құрылтайшы қажет.",
]


embeddings = model.encode(
    texts,
    normalize_embeddings=True,
)


print("Embedding shape:", embeddings.shape)

print(
    "First embedding norm:",
    (embeddings[0] @ embeddings[0]) ** 0.5,
)
