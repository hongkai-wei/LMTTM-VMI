# [LMTTM-VMI: Linked Memory Token Turing Machine for 3D Volumetric Medical Image Classification](https://hongkai-wei.github.io/LMTTM-VMI_Page/).


Biomedical imaging is vital for the diagnosis and treatment of various medical conditions, yet the effective integration of deep learning technologies into this field presents challenges. Traditional methods often struggle to efficiently capture the spatial characteristics and intricate structural features of 3D volumetric medical images, limiting memory utilization and model adaptability. 
To address this, we introduce a **L**inked **M**emory **T**oken **T**uring **M**achine (**LMTTM**), which utilizes external linked memory to efficiently process spatial dependencies and structural complexities within 3D volumetric medical images, aiding in accurate diagnoses. LMTTM can efficiently record the features of 3D volumetric medical images in an external linked memory module, enhancing complex image classification through improved feature accumulation and reasoning capabilities. Our experiments on six 3D volumetric medical image datasets from the MedMNIST v2 demonstrate that our proposed LMTTM model achieves average ACC of 82.4\%, attaining state-of-the-art (SOTA) performance. Moreover, ablation studies confirmed that the Linked Memory outperforms its predecessor, TTM's original Memory, by up to 5.7\%, highlighting LMTTM's effectiveness in 3D volumetric medical image classification and its potential to assist healthcare professionals in diagnosis and treatment planning.

------

- [LMTTM-VMI: Linked Memory Token Turing Machine for 3D Volumetric Medical Image Classification](#lmttm-vmi)
  - [Results](#results)
  - [Protocol](#protocol)
  - [LMTTM-Architecture](#lmttm-architecture)
    - [Linked Memory](#test)
    - [Read from Linked Memory](#read)
    - [Memory Distillation Unit](#mdu)
    - [Write to Linked Memory](#write)
  - [Requirement](#requirement)
  - [Preparation](#preparation)
  - [MedMNIST3D in MedMNIST v2](#medmnist3d)
    - [Train](#train)
    - [Test](#test)
  - [Acknowledge](#acknowledge)
  - [License](#license)

------

## Results

> LMTTM-VMI on AdrenalMNIST3D and OrganMNIST3D dataset 

![result0](./images/result0.gif)

> LMTTM-VMI on FractureMNIST3D and NoduleMNIST3D dataset 

![result1](./images/result1.gif)

> LMTTM-VMI on SynapseMNIST3D and VesselMNIST3D dataset 

![result2](./images/result2.gif)

## Protocol

- <img src="https://latex.codecogs.com/svg.image?\textrm{M}^{t-1},\textrm{M}^{t},\textrm{M}^{t&plus;1}"/> represents memory blocks at different times, the three moments of memory are the past moment, this moment, and the next moment.
- <img src="https://latex.codecogs.com/svg.image?\textrm{I}^{t}"/> represents the input at moment t.
- <img src="https://latex.codecogs.com/svg.image?\textrm{O}^{t}"/> represents the output features obtained after MDU output. It is used for result classification and writing to memory.
- <img src="https://latex.codecogs.com/svg.image?\textrm{W},\textrm{N}"/> represents the size of the memory block, respectively the memory block length and dimension.
- <img src="https://latex.codecogs.com/svg.image?\textrm{Z}^{t-1},\textrm{Z}^{t},\textrm{Z}^{t&plus;1}"/> denotes the feature information read from memory at different moments in time.

## LMTTM-Architecture

Schematics of the Linked Memory Token Turing Machine (LMTTM). On the left, the network's processing at time stamp $t$ is depicted, where the current memory block <img src="https://latex.codecogs.com/svg.image?\textrm{M}^{t}"/> and its adjacent blocks <img src="https://latex.codecogs.com/svg.image?\textrm{M}^{t-1}"/> and <img src="https://latex.codecogs.com/svg.image?\textrm{M}^{t+1}"/> are synergistically processed by the Tri-Temporal Memory Collaborative (T-TMC) module, which we will refer to as Simplified Memory. Simplified memory, along with preprocessed input tokens, is read through the Read module, and the extracted information is then refined into more efficient output tokens by the Memory Distillation Unit(MDU) for image classification. The simplified Memory, input tokens, and output tokens are then written in the subsequent Memory block <img src="https://latex.codecogs.com/svg.image?\textrm{M}^{t+1}"/> via the Write module, completing the LMTTM cycle for time stamp $t$ and facilitating a linked cyclic update of the Memory content. The right side shows a similar process for time stamp $t+1$.
![framework](./images/framework.png)

### Linked Memory

Illustration of different memory. (a) is the memory structure of TTM, where the
entire memory block is iteratively updated during the interaction. (b) is the memory
structure of our LMTTM, which can be viewed as a chain of k memory units linked
together, and is iteratively updated one memory unit at a time during the interaction.

![linkedmemory](./images/linkedmemory.png)

### Read from Linked Memory

Illustration of different Reads. (a) is the entire block of memory that will be read by the TTM. (b) is the coordinated read operation for the current moment $t$, the past moment $t-1$ and the future moment $t+1$.
![read](./images/read.png)

### Memory Distillation Unit

In the LMTTM architecture, the processing unit is designed as <img src="https://latex.codecogs.com/svg.image?\textrm{O}^{t}=\textrm{MDU}(\textrm{P}^{t})"/>, which receives the tokens $3r$ obtained from the read operation <img src="https://latex.codecogs.com/svg.image?\textrm{P}^{t}"/> and processes them. The MDU outputs a set of vectors <img src="https://latex.codecogs.com/svg.image?\textrm{O}^{t}"/> containing the $3r$ tokens, which are used in subsequent write operations and in generating predictions in the 3D volumetric medical image classification task. For tasks that require predictions at each time stamp, we introduce a fully connected layer after <img src="https://latex.codecogs.com/svg.image?\textrm{O}^{t}"/> to enhance the classification capability of the model, which is implemented as <img src="https://latex.codecogs.com/svg.image?\textrm{Y}^{t}=\textrm{Classifier}(\textrm{P}^{t})=\textrm{W}_{o}\textrm{O}^{t}"/>, where <img src="https://latex.codecogs.com/svg.image?\textrm{W}_{o}"/> is the weight matrix of the fully connected layer used to map <img src="https://latex.codecogs.com/svg.image?\textrm{O}^{t}"/> to the input space of the classifier. This produces the final classification prediction.

### Write to Linked Memory

Illustration of different Writes. (a) is the entire block of memories from the current moment in TTM being written to the next moment in the entire block. (b) is the entire block of tri-temporal memories from the current moment $t$, the past moment $t-1$, and the future moment $t+1$ being synergized and written to the future moment $t+1$ in LMTTM.
![write](./images/write.png)

## Requirement

| Name  |Version   |
| ------------ | ------------ |
|  Python | 3.8  |
|  CUDA | >=10.1  |
|  Pytorch | 1.12.1  |

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
## MedMNIST3D in MedMNIST v2
[MedMNIST v2](https://github.com/MedMNIST/MedMNIST) contains a collection of six preprocessed 3D volumetric medical image datasets. It is designed to be educational, standardized, diverse, and lightweight and can be used as a general classification benchmark in 3D volumetric medical image analysis.  
The program will automatically download the corresponding dataset at runtime.

 For different datasets and tasks, different parameters need to be configured.
The parameters are configured in `<path>\config\base.json`.
To run the program, you need to change the `dataset_name` field in `base.json` to the name of the corresponding dataset, which can be OrganMNIST3D, SynapseMNIST3D, AdrenalMNIST3D, FractureMNIST3D, NoduleMNIST3D, VesselMNIST3D, and also in `out_class_num` you need to change the number of classes in the corresponding dataset. VesselMNIST3D, also in `out_class_num` you need to modify the number of classes in the corresponding dataset.


### Train
Modify the `dataset_name`, `out_class_num` and `epoch` parameters in the configuration file `<path>\config\base.json`.  
And then activate your virtual environment, followed by executing `python train.py base.json`. 

###  Test
As with the TRAIN process, modify the corresponding parameters, then activate the virtual environment and execute `python predict.py base.json`

## Acknowledge
This work is based on [TTM(Token Turing Machine)](https://arxiv.org/abs/2211.09119) and inspired by [CMN (Collaborative Memory Network)](https://ieeexplore.ieee.org/document/9264159).

## License

LMTTM's code is released under the Apache License 2.0. A permissive license whose main conditions require preservation of copyright and license notices. Contributors provide an express grant of patent rights. Licensed works, modifications, and larger works may be distributed under different terms and without source code.See [LICENSE](https://github.com/WHKai666666/LMTTM-VMI/blob/main/LICENSE "LICENSE") for further details. 

------
