# VTON-SCFA
The official implementation of the paper "VTON-SCFA: A Virtual Try-On Network Based on the Semantic Constraint and Flow Aligning".

<br/><br/>

![VTON-SCFA](./teaser.png "Teaser PNG")
## Requirements
- python 3.6
- pytorch 1.0.0
- torchvision 0.3.0
- cuda 10.0
- opencv

To install requirements:
```setup
conda create -n dcton python=3.6
conda activate dcton
conda install pytorch==1.0.10 torchvision==0.3.0 cuda100
pip install tensorboardX
pip install opencv-python
pip install imdb
pip install tqdm
```

# Usage #
Clone the repo and install requirements through ```pip install -r requirements.txt``` 

## Data Processing
Dataset download instructions and link of dataset can be found from official repo of [CP-VTON](https://github.com/sergeywong/cp-vton) and [VITON](https://github.com/xthan/VITON) </br>
Put dataset in `data` folder



## License
The use of this code is restricted to non-commercial research.

## Acknowledgement 
Thanks for [levindabhi-SieveNet](https://github.com/levindabhi/SieveNet) for providing the useful codes.
