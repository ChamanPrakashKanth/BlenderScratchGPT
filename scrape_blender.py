import os
from bs4 import BeautifulSoup

INPUT_FOLDER = "blender_api_5_1"
OUTPUT_FILE = "blender_api_5_1.txt"

html_files = []

# Find every HTML file
for root, dirs, files in os.walk(INPUT_FOLDER):
    for file in files:
        if file.endswith(".html"):
            html_files.append(
                os.path.join(root, file)
            )

print(f"Found {len(html_files)} HTML files")


with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for i, filepath in enumerate(html_files, 1):

        print(f"[{i}/{len(html_files)}] Processing: {filepath}")

        try:
            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as f:
                html = f.read()

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            # Remove things that are not documentation
            for tag in soup([
                "script",
                "style",
                "nav",
                "footer",
                "header"
            ]):
                tag.decompose()

            # Prefer the actual documentation body
            content = (
                soup.find("main")
                or soup.find(
                    "div",
                    class_="document"
                )
                or soup.body
            )

            if content is None:
                continue

            # Get page title
            title = soup.title

            if title:
                title_text = title.get_text(
                    " ",
                    strip=True
                )
            else:
                title_text = os.path.basename(filepath)

            # Write document separator
            out.write("\n\n")
            out.write("=" * 100)
            out.write("\n")
            out.write(title_text)
            out.write("\n")
            out.write("=" * 100)
            out.write("\n\n")

            # Preserve code examples
            for code in content.find_all("pre"):

                code_text = code.get_text(
                    "\n",
                    strip=True
                )

                if code_text:
                    code.replace_with(
                        "\n\nCODE:\n"
                        + code_text
                        + "\n\n"
                    )

            # Extract text
            text = content.get_text(
                separator="\n",
                strip=True
            )

            # Clean excessive blank lines
            lines = []

            for line in text.splitlines():

                line = line.strip()

                if not line:
                    continue

                lines.append(line)

            clean_text = "\n".join(lines)

            out.write(clean_text)
            out.write("\n")

        except Exception as exc:
            print(f"Error processing {filepath}: {exc}")


print("Saved:", OUTPUT_FILE)