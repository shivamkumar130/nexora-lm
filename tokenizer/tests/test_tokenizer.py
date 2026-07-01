from tokenizer.tokenizer import encode, decode

text = "Flexible Respiration Sensor"

tokens = encode(text)

print(tokens)

print(
    decode(tokens)
)