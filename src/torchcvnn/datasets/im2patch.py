# MIT License

# Copyright (c) 2024 Xuan-Huy Nguyen

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


import os, tqdm, glob, json, shutil
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union


def check_path(path: str) -> None:
    """Check if a path exist. If not, create it. Else, remove all file and folders inside of it.

    Args:
        path (str): Path to check
    """
    
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for file in glob.glob(os.path.join(path, '*')):
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)


class ImagetoPatch:
    
    def __init__(
        self, 
        data_dir: str, 
        patch_size: int | Tuple, 
        stride: int | Tuple, 
        offset: float = np.spacing(1),
        normalize_min_max: bool = True
    ) -> None:
        """Module for splitting image to patch, and save image folder statistics.

        Args:
            data_dir (str): parent directory of the images
            patch_size (int | Tuple): size of patch
            stride (int | Tuple): size of the sliding window
            offset (float, optional): Smallest value adding to data to avoid -infinity. Defaults to np.spacing(1).
            normalize_min_max (bool, optional): normalization option. Defaults to True.
        """
        self.data_dir = data_dir
        self.save_dir = data_dir + '_split'
        check_path(self.save_dir)
        
        self.patch_size =  patch_size
        self.stride = stride
        self.offset = offset
        
        self.get_metadata(normalize_min_max)
        
    def load_npy(self, data_path: str) -> Tuple[np.ndarray, str]:
        """Load .npy file.

        Args:
            data_path (str): File path

        Raises:
            TypeError: Unable to load file format different than .npy and .slc

        Returns:
            Tuple[np.ndarray, str]: Data in numpy array format and its filename
        """
        
        filename = os.path.basename(data_path)
        action_dict = {
            '.slc': np.fromfile(data_path, dtype=np.complex64),
            '.npy': np.load(data_path)
        }
        
        if os.path.splitext(data_path)[1] not in action_dict.keys():
            raise TypeError(f'cannot load file {data_path}')
        
        for (k,v) in action_dict.items():
            if os.path.splitext(filename)[1] == k:
                return v, filename
            
    def split_sar_data(
        self,
        data: np.ndarray, 
        filename: str,
    ) -> None:
        """Split SAR image into multiple patches

        Args:
            data (np.ndarray): SAR image 
            filename (str): SAR image filename
            savedir (str): Save directory
            size (int | Tuple): Patch size
            stride (int | Tuple): Size of the crop window
        """
        
        if isinstance(self.patch_size, Tuple):
            size_x, size_y = self.patch_size[0], self.patch_size[1]
            stride_x, stride_y = self.stride[0], self.stride[1]
        else:
            size_x = size_y = self.patch_size
            stride_x = stride_y = self.stride

        with tqdm.tqdm(
            total=len(range(0, data.shape[0] - size_x, stride_x)) * len(range(0, data.shape[1] - size_y, stride_y))
        ) as pbar:
            for l in range(0, data.shape[0] - size_x, stride_x):
                for m in range(0, data.shape[1] - size_y, stride_y):
                    save_filename = os.path.splitext(filename)[0] + f'_{round(l)}_{round(m)}.npy'
                    np.save(Path(os.path.join(str(self.save_dir), save_filename)), data[l:l + size_x, m:m + size_y])
                    pbar.update(1)
                    
    def get_min_max(self, data: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Get minimum and maximum value of the image

        Args:
            data (np.ndarray): Image in np.ndarray format

        Returns:
            Dict[str, Union[np.ndarray, float]]: normalized data, min and max of the original data
        """

        data = np.log(np.abs(data) + self.offset)
        data_min, data_max = np.min(data), np.max(data)
        data = (data - data_min) / (data_max - data_min)
        return {
            'data': data,
            'min': data_min,
            'max': data_max
        }
     
    def get_mean_std(self, data: np.ndarray) -> Dict[str, Union[np.ndarray, complex | float]]:
        """Get mean and standard deviation value of the image

        Args:
            data (np.ndarray): Image in np.ndarray format

        Returns:
            Dict[str, Union[np.ndarray, complex | float]]: normalized data, mean and std of the original data.
        """
        
        data_mean, data_std = np.mean(data), np.std(data)
        data = (data - data_mean) / (data_std)
        
        return {
            'data': data,
            'mean': data_mean,
            'std': data_std
        }
                    
    def get_metadata(self, normalize_min_max: bool) -> None:
        """Compute the image folder's statistics, normalize image and split in into patches.

        Args:
            normalize_min_max (bool): Whether normalize image in min-max scale or with mean 0 and std 1.
        """
        metadata_dict = {}
        
        for file in glob.glob(f'{self.data_dir}/*.npy'):
            data, filename = self.load_npy(file)
            data_min_max = self.get_min_max(data)
            data_mean_std = self.get_mean_std(data)
            metadata_dict[filename] = {
                'min': str(data_min_max['min']),
                'max': str(data_min_max['max']),
                'mean': str(data_mean_std['mean']),
                'std': str(data_mean_std['std'])
            }
            
            if normalize_min_max:
                data = data_min_max['data']
            else:
                data = data_mean_std['data']
            
            self.split_sar_data(data, filename)

        metadata_dict['folder_min'] = np.min([float(v['min']) for v in metadata_dict.values()])
        metadata_dict['folder_max'] = np.max([float(v['max']) for v in metadata_dict.values() if isinstance(v, Dict)])
            
        with open(os.path.join(self.save_dir, 'metadata.json'), 'w') as json_file:
            json.dump(metadata_dict, json_file, indent=4)