# [LMTTM-VMI: Linked Memory Token Turing Machine for 3D Volumetric Medical Image Classification](https://whkai666666.github.io/LMTTM-VMI_Page/)

Biomedical imaging is vital for the diagnosis and treatment of various medical conditions, yet the effective integration of deep learning technologies into this field presents challenges. Traditional methods often struggle to efficiently capture the spatiotemporal characteristics of image sequences, limiting memory utilization and model adaptability. 
To address this, we introduce a **L**inked **M**emory **T**oken **T**uring **M**achine (**LMTTM**), which utilizes external linked memory to efficiently process spatial-temporal dependencies in 3D volumetric medical images, aiding in accurate diagnoses. **LMTTM** can efficiently record the features of 3D volumetric medical images in an external linked memory module, enhancing complex image classification through improved feature accumulation and reasoning capabilities.
Our experiments on MedMNIST v2 datasets demonstrate the superiority of **LMTTM** over state-of-the-art models and its predecessor TTM, showcasing its potential to transform medical image interpretation and assist healthcare professionals in diagnosis and treatment planning.

------

- [LMTTM-VMI: Linked Memory Token Turing Machine for 3D Volumetric Medical Image Classification](#lmttm-vmi)
  - [Results](#results)
  - [RoadMap](#roadmap)
  - [Protocol](#protocol)
  - [LMTTM-Architecture](#lmttm-architecture)
    - [Linked Memory](#test)
    - [Read from Linked Memory](#read)
    - [Memory Distillation Unit](#mdu)
    - [Write to Linked Memory](#write)
  - [Requirement](#requirement)
  - [Preparation](#preparation)
  - [MedMNIST3D in MedMNIST v2](#MedMNIST3D)
    - [Test](#test)
    - [Train](#train)
  <!-- - [Citation](#citation) -->
  - [Acknowledge](#acknowledge)
  - [License](#license)


## Requirement
| Name  |Version   |
| ------------ | ------------ |
|  CUDA | >=10.1  |
|  Pytorch | 1.12.1  |
|  Python | 3.8  |

The rest of the environment is installed with the following command
```shell
cd <project path>
pip install -r requirement.txt
```
## Preparation
Clone the repository
```shell
git clone <repository url>
```
## MedMNIST3D
[MedMNIST v2](https://github.com/MedMNIST/MedMNIST) contains a collection of six preprocessed 3D volumetric medical image datasets. It is designed to be educational, standardized, diverse, and lightweight and can be used as a general classification benchmark in 3D volumetric medical image analysis.  
The program will automatically download the corresponding dataset at runtime.

 For different datasets and tasks, different parameters need to be configured.
The parameters are configured in `<path>\config\base.json`.
To run the program, you need to change the `dataset_name` field in `base.json` to the name of the corresponding dataset, which can be organmnist3d, synapsemnist3d, adrenalmnist3d, fracturemnist3d, nodulemnist3d, vesselmnist3d, and also in `out_class_num` you need to change the number of classes in the corresponding dataset. vesselmnist3d, also in `out_class_num` you need to modify the number of classes in the corresponding dataset.


- ### Train
Modify the `dataset_name`, `out_class_num` and `epoch` parameters in the configuration file `<path>\config\base.json`.  
And then activate your virtual environment, followed by executing `python train.py base.json`. 

- ###  Test
As with the TRAIN process, modify the corresponding parameters, then activate the virtual environment and execute `python predict.py base.json`

## Acknowledge
This work is based on [TTM(Token Turing Machine)](https://arxiv.org/abs/2211.09119) and inspired by [CMN (Collaborative Memory Network)](https://ieeexplore.ieee.org/document/9264159).

## License

LMTTM's code is released under the Apache License 2.0. A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.See [LICENSE](http://https://github.com/WHKai666666/LMTTM-VMI/blob/main/LICENSE "LICENSE") for further details. 

------


