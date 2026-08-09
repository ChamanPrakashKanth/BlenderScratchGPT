import re

INPUT_FILE = "data/blender_docs.txt"
OUTPUT_FILE = "data/blender_clean.txt"

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


# Normalize Windows line endings
text = text.replace("\r\n", "\n")

# Remove excessive whitespace
text = re.sub(
    r"[ \t]+",
    " ",
    text
)

# Remove excessive blank lines
text = re.sub(
    r"\n{3,}",
    "\n\n",
    text
)

text = text.strip()


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(text)


print("Characters:", len(text))
print("Saved:", OUTPUT_FILE)