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

import logging
import pathlib
from typing import Any, Callable, Literal, Optional, Union

import scipy
import torch
from torch.utils.data import Dataset

from .parse_xml import xml_to_dict


def gather_ATRNetSTAR_datafiles(
    rootdir: pathlib.Path,
    split: Literal["train", "test", "all"],
) -> list[str]:
    """
    This function gathers all the ATRNet-STAR datafiles from the specified split in the root directory

    It returns a list of the paths of the samples (without the extension).
    """

    data_files = []

    if split != "all":
        split_dir = rootdir / split
    else:
        split_dir = rootdir
    logging.debug(f"Looking for all samples in {split_dir}")

    # only look for data files (avoid duplicates)
    for filename in split_dir.glob("**/*.mat"):
        if not filename.is_file():
            continue

        # strip file of the .xml or .mat extension
        sample_name = str(filename.with_suffix(""))

        # add sample name to the list of known samples
        data_files.append(sample_name)

    return data_files


class ATRNetSTARSample:
    """
    This class implements a sample from the ATRNet-STAR dataset.
    Only slant-range quad polarization complex images are supported.

    The extracted complex image is stored in the `data` attribute.

    The data is a dictionnary, keys being the polarization ('HH', 'HV', 'VH' or 'VV')
    and values a numpy array of the corresponding complex image.

    The image annotations are stored in the `annotation` attribute. It contains all the fields
    of the XML annotation, in a dictionnary.

    Arguments:
        sample_name (str): the name of the file to load WITHOUT the .mat or .xml extension
        transform (Callable): (optional) A transform to apply to the data
    """

    DATA_FILETYPE = ".mat"
    HEADER_FILETYPE = ".xml"

    POL_KEY_TRANSLATION_DICT = {
        "HH": "data_hh",
        "HV": "data_hv",
        "VH": "data_vh",
        "VV": "data_vv",
    }

    def __init__(self, sample_name: str, transform: Optional[Callable] = None):
        self._annotation = xml_to_dict(sample_name + self.HEADER_FILETYPE)
        self._data = {}
        self.transform = transform

        image = scipy.io.loadmat(sample_name + self.DATA_FILETYPE)
        for pol in self.POL_KEY_TRANSLATION_DICT.keys():
            self._data[pol] = image[self.POL_KEY_TRANSLATION_DICT[pol]]

    @property
    def data(self):
        if self.transform is not None:
            return self.transform(self._data)
        return self._data

    @property
    def annotation(self):
        return self._annotation


class ATRNetSTAR(Dataset):
    """
    Implements a PyTorch Dataset for the ATRNet-STAR dataset presented in :

    Yongxiang Liu, Weĳie Li, Li Liu, Jie Zhou, Bowen Peng, Yafei Song, Xuying Xiong, Wei Yang,
    Tianpeng Liu, Zhen Liu, & Xiang Li. (2025).
    ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild.

    Only slant-range quad polarization complex images are supported.

    The dataset is composed of pre-defined benchmarks (see paper).
    Dowloading them automatically is possible, but a Hugging Face authentification token
    is needed as being logged in is required for this dataset.

    Warning : samples are ordered by type, shuffling them is recommended.

    Arguments:
        root_dir (str): The root directory in which the different benchmarks are placed.
                        Will be created if it does not exist.
        benchmark (str): (optional) Chosen benchmark. If not specified, SOC_40 (entire dataset) will be used as default.
        split (str): (optional) Chosen split ('train', 'test' or 'all' for both). Those are pre-defined by the dataset. Default: 'all'
        download (bool): (optional) Whether or not to download the dataset if it is not found. Default: False
        class_level (str): (optional) The level of precision chosen for the class attributed to a sample.
                            Either 'category', 'class' or 'type'. Default: 'type' (the finest granularity)
        get_annotations (bool): (optional) If `False`, a dataset item will be a tuple (`sample`, `target class`) (default).
                                If `True`, the entire sample annotation
                                dictionnary will also be returned: (`sample`, `target class`, `annotation dict`).
        transform (Callable): (optional) A transform to apply to the data
    """

    # Hugging Face repository constants
    HF_REPO_ID = "waterdisappear/ATRNet-STAR"
    HF_BENCHMARKS_DIR_PATH = pathlib.Path("Slant_Range/complex_float_quad/")

    # SOC_40classes is the complete original dataset, with the train/test splits defined
    BENCHMARKS = [
        "SOC_40classes",
        "EOC_azimuth",
        "EOC_band",
        "EOC_depression",
        "EOC_scene",
    ]

    _ALLOWED_BENCHMARKS = BENCHMARKS + ["SOC_40"]
    # prettier logs later
    _ALLOWED_BENCHMARKS.sort(reverse=True)

    # EOC_polarization consists of training using one polarization and testing using another
    # (to implement ? or leave it to the user ?)
    #
    # SOC_50 mixes the MSTAR dataset with a similar amount of samples from ATRNet-STAR.
    # This should probably be done manually by the user
    #

    ### class names for all levels

    CATEGORIES = ["Car", "Speacial", "Truck", "Bus"]

    CLASSES = [
        "Large_Car",
        "Medium_SUV",
        "Compact_SUV",
        "Mini_Car",
        "Medium_Car",
        "ECV",
        "Ambulance",
        "Road_Roller",
        "Shovel_Loader",
        "Light_DT",
        "Pickup",
        "Mixer_Truck",
        "Heavy_DT",
        "Medium_TT",
        "Light_PV",
        "Heavy_FT",
        "Forklift",
        "Heavy_ST",
        "Small_Bus",
        "Medium_Bus",
        "Large_Bus",
    ]

    TYPES = [
        "Great_Wall_Voleex_C50",
        "Hongqi_h5",
        "Hongqi_CA7180A3E",
        "Chang'an_CS75_Plus",
        "Chevrolet_Blazer_1998",
        "Changfeng_Cheetah_CFA6473C",
        "Jeep_Patriot",
        "Mitsubishi_Outlander_2003",
        "Lincoln_MKC",
        "Hawtai_EV160B",
        "Chery_qq3",
        "Buick_Excelle_GT",
        "Chery_Arrizo 5",
        "Lveco_Proud_2009",
        "JINBEI_SY5033XJH",
        "Changlin_8228-5",
        "SDLG_ZL40F",
        "Foton_BJ1045V9JB5-54",
        "FAW_Jiabao_T51",
        "WAW_Aochi_1800",
        "Huanghai_N1",
        "Great_Wall_poer",
        "CNHTC_HOWO",
        "Dongfeng_Tianjin_DFH2200B",
        "WAW_Aochi_Hongrui",
        "Dongfeng_Duolika",
        "JAC_Junling",
        "FAW_J6P",
        "SHACMAN_DeLong_M3000",
        "Hyundai_HLF25_II",
        "Dongfeng_Tianjin_KR230",
        "SHACMAN_DeLong_X3000",
        "Wuling_Rongguang_V",
        "Buick_GL8",
        "Chang'an_Starlight_4500",
        "Dongfeng_Forthing_Lingzhi",
        "Yangzi_YZK6590XCA",
        "Dongfeng_EQ6608LTV",
        "MAXUS_V80",
        "Yutong_ZK6120HY1",
    ]

    def __init__(
        self,
        root_dir: str,
        benchmark: Optional[str] = None,
        split: Literal["train", "test", "all"] = "all",
        download: bool = False,
        class_level: Literal["type", "class", "category"] = "type",
        get_annotations: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.class_level = class_level
        self.download = download
        self.get_annotations = get_annotations
        self.transform = transform

        if benchmark is None:
            # if no benchmark is given, default behavior should be to use the entire dataset (SOC_40)
            logging.info(
                "No benchmark was specified. SOC_40 (full dataset) will be used."
            )
            benchmark = "SOC_40classes"

        self.benchmark = benchmark
        self._verify_inputs()

        if self.benchmark == "SOC_40":
            # allow use of the name given to the benchmark in the paper instead of the file
            # name actually used in their repository
            # (more consistent with the rest of their file naming)
            self.benchmark = "SOC_40classes"

        self.benchmark_path = self.root_dir / self.benchmark

        if not self.benchmark_path.exists():
            if not download:
                raise RuntimeError(
                    f"{self.benchmark} benchmark not found. You can use download=True to download it"
                )
            else:
                self._download_dataset()

        # gather samples
        self.datafiles = gather_ATRNetSTAR_datafiles(
            rootdir=self.benchmark_path, split=self.split
        )
        logging.debug(f"Found {len(self.datafiles)} samples.")

    def _verify_inputs(self) -> None:
        """Verify inputs are valid"""
        if self.class_level not in ["type", "class", "category"]:
            raise ValueError(
                f"Unexpected class_level value. Got {self.class_level} instead of 'type', 'class' or 'category'."
            )

        if self.benchmark not in self._ALLOWED_BENCHMARKS:
            benchmarks_with_quotes = [f"'{b}'" for b in self._ALLOWED_BENCHMARKS]
            raise ValueError(
                f"Unknown benchmark. Should be one of {', '.join(benchmarks_with_quotes)} or None"
            )

        if self.split not in ["train", "test", "all"]:
            raise ValueError(
                f"Unexpected split value. Got {self.split} instead of 'train', 'test' or 'all'."
            )

    def _download_dataset(self) -> None:
        """
        Downloads the specified benchmark.
        Will be placed in a directory named like the benchmark, in root_dir
        """
        from .download import check_7z, download_benchmark

        check_7z()
        download_benchmark(
            benchmark=self.benchmark,
            root_dir=self.root_dir,
            hf_repo_id=self.HF_REPO_ID,
            hf_benchmark_path=self.HF_BENCHMARKS_DIR_PATH,
        )

    @property
    def classes(self) -> list[str]:
        """
        Get the names of all classes at class_level,
        """
        match self.class_level:
            case "category":
                return self.CATEGORIES

            case "class":
                return self.CLASSES

            case "type":
                return self.TYPES

            case _:
                raise ValueError(
                    f"Unexpected class_level value. Got {self.class_level} instead of type, class or category."
                )

    def __len__(self) -> int:
        return len(self.datafiles)

    def __getitem__(self, index: int) -> tuple:
        """
        Returns the sample at the given index. Applies the transform
        if provided. If `get_annotations` is True, also return the entire annotation dict.
        The return type depends on the transform applied and whether or not get_annotations is True

        Arguments:
            index : index of the sample to return

        Returns:
            data : the sample
            class_idx : the index of the class in the classes list
            (Optional) annotation : the annotation dict of the sample. Only if `get_annotations` is True.

        """
        sample_name = self.datafiles[index]
        sample = ATRNetSTARSample(sample_name=sample_name, transform=self.transform)
        class_idx = self.classes.index(sample.annotation["object"][self.class_level])

        if self.get_annotations:
            return (
                sample.data,
                class_idx,
                sample.annotation,
            )

        return sample.data, class_idx
