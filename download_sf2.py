import urllib.request
import os

url = "https://sourceforge.net/p/mscore/code/HEAD/tree/trunk/mscore/share/sound/TimGM6mb.sf2?format=raw"
output_path = "assets/TimGM6mb.sf2"

print(f"Downloading {url} to {output_path}...")
urllib.request.urlretrieve(url, output_path)
print("Download complete.")
