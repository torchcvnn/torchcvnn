# MIT License

# Copyright (c) 2025 Rodolphe Durand

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

import glob
import os
import pathlib
import platform
import shutil
import subprocess

import huggingface_hub


def check_7z():
    """
    Checks if 7zip is installed and if the command is available. If not, raises an error asking the user to install it.
    """
    if shutil.which("7z") is None:
        message = "7zip is needed to unpack the dataset files. Please install it and add command to path."
        if platform.system() == "Linux":
            message += " The following command can be used :\n\n     sudo apt-get install p7zip-full\n"

        raise RuntimeError(message)


def download_benchmark(
    benchmark: str,
    root_dir: pathlib.Path,
    hf_repo_id: str,
    hf_benchmark_path: pathlib.Path,
):
    """
    Download the specified benchmark in the given root directory.

    The dowload uses Hugging Face. The user will be prompted to log in using an acess token
    A Hugging Face account is thus required to download the dataset (even when doing so manually)
    """
    huggingface_hub.login(new_session=False)

    # create root_dir if it does not exist
    root_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(root_dir.absolute())

    # gather all volumes of the split 7z archive (file.7z.001, file.7z.002, ...)
    benchmark_glob = f"{str(hf_benchmark_path / benchmark)}.7z.*"
    huggingface_hub.snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=benchmark_glob,
        local_dir=root_dir.absolute(),
        repo_type="dataset",
    )

    # downloaded files maintain their original file structure
    # move them back to root_dir and delete empty directories
    for zip_file in (root_dir / hf_benchmark_path).glob("*.*"):
        shutil.move(zip_file, root_dir)
    os.removedirs(hf_benchmark_path)

    # unzip the first archive (7zip automatically joins all volumes)
    subprocess.run(f"7z x {benchmark}.7z.001", shell=True)

    # delete unpacked 7zip files
    for f in glob.glob(f"{benchmark}.7z.*"):
        os.remove(f)
