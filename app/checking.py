from pypdf import PdfReader

reader = PdfReader("../data/raw/general/Department_CS_Complete_Info.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Search for HoD info
if "madhu" in text.lower():
    print("Found 'Madhu' in PDF")
    # Print context around it
    idx = text.lower().find("madhu")
    print(text[max(0, idx-100):idx+200])
else:
    print("'Madhu' NOT found in PDF")

# Check for "head"
if "head" in text.lower():
    print("\nFound 'head' in PDF")
else:
    print("\n'Head' NOT found")