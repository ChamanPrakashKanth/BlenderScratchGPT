from generate import generate


print("Blender GPT")
print("Type 'exit' to quit.")
print()


while True:

    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    response = generate(
        prompt,
        max_new_tokens=200,
        temperature=0.8
    )

    print()
    print("AI:", response)
    print()