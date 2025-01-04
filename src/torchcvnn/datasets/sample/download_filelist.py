# MIT License

# Copyright (c) 2024 Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
import requests
from bs4 import BeautifulSoup
import re

SAMPLE_base_link = "https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public/blob/refs/heads/master/mat_files/"

SAMPLE_classes = [
    "2s1",
    "bmp2",
    "btr70",
    "m1",
    "m2",
    "m35",
    "m548",
    "m60",
    "t72",
    "zsu23",
]


def generate_file_list():
    """
    Download the list of mat files available for every mode and every class
    and save them as a dictionnary to be loaded by torchcvnn

    We do this "asynchronously" because it happens, from time to time, that
    requests is returning an empty list of files. Requesting and parsing is anyway
    some overhead we can avoid at runtime.
    """
    with open("filelist.py", "w") as fh:
        fh.write("filelist = {\n")
        for mode in ["real", "synth"]:
            fh.write(f'"{mode}": {{\n')
            for cl in SAMPLE_classes:
                url = f"{SAMPLE_base_link}{mode}/{cl}/"
                result = requests.get(url)
                soup = BeautifulSoup(result.text, "html.parser")
                matfiles = soup.find_all(title=re.compile("\.mat$"))

                fh.write(f'"{cl}": [\n')
                fh.write(",\n".join([f'\t"{f.get("title")}"' for f in matfiles]))
                fh.write("],\n")

                if len(matfiles) == 0:
                    # print("=====================================\n")
                    print(f"Empty list of files for {cl} in {mode}")
                    # print(result.text)
                    print(
                        "============================================================\n"
                    )

            fh.write("},\n")
        fh.write("}\n")


if __name__ == "__main__":
    generate_file_list()
