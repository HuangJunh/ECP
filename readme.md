# ECP (Evolving Comprehensive Proxies) 
Junhao Huang, Bing Xue, Yanan Sun, and Mengjie Zhang. 2025. Evolving Comprehensive Proxies for Zero-Shot Neural Architecture Search. 
In Genetic and Evolutionary Computation Conference (GECCO '25), July 14-18, 2025, Malaga, Spain. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3712256.3726315

📑 [Read the Paper]()


## Preparation
This code is tested with Python 3.12.7, PyTorch 2.5.1, and CUDA 12.7. 

- Download datasets (CIFAR-10, CIFAR-100, ImageNet-16-120) from https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4 and put them in `./datasets`
- Download benchmark datasets and put them in `./APIs`
    - NATS-Bench-TSS (NAS-Bench-201): https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view
    - NATS-Bench-SSS: https://drive.google.com/file/d/1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA/view
    - NDS: https://dl.fbaipublicfiles.com/nds/data.zip

## Proxy Search
- Conduct proxy search on NATS-Bench by running `sh run.sh` 


## Citation
If you use this code in your research, please cite the following paper:
```bibtex
@inproceedings{ECP,
  title={Evolving Comprehensive Proxies for Zero-Shot Neural Architecture Search},
  author={Huang, Junhao and Xue, Bing and Sun, Yanan and Zhang, Mengjie and Yen, Gary G.},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={},
  year={2025}
}
```