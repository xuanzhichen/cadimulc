# Hybrid-Based Causal Discovery by Cadimulc
<div  align="center"> 
<img src="cadimulc_logo.png" align=center />
</div>

Note: This repository is still in building (03-31). The first release will be done in the next few days.

CADIMULC is a Python package standing for the task: **CA**usal **DI**scovery 
with **M**ultiple **L**atent **C**onfounders, providing easy-to-use light APIs 
to learn an empirical causal graph from generally raw data with relatively efficiency.

The package integrates implementations of **hybrid-based approaches** involving the popular [MLC-LiNGAM algorithm](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Causal+discovery+in+linear+non-gaussian+acyclic+model+with+multiple+latent+confounders&btnG=),
along with the "micro" **workflow of causal discovery**, such as data generation, learning results evaluation, and graphs visualization.

For more information in the following.
* **Documentation**: https://xuanzhichen.github.io/cadimulc/
* **Paper Presentation, 2024**
  *  **YouTube**: https://www.youtube.com/channel/UC0CenFxAC9yP5UcnZzJ-cyw
  *  **Bilibili**: https://www.youtube.com/channel/UC0CenFxAC9yP5UcnZzJ-cyw

## Overview
### The Hybrid Methodology

Write down the descriptions here.
<div  align="center"> 
<img src="hybrid_methodology.png" width = "850" align=center />
</div>

### Causal Discovery Workflow
Write down the descriptions here.

### Modules in Cadimulc
Write down the descriptions here.

## Citation
Please cite the following paper(s) depending on which approach you use in your reports or publications:
```
@article{chen2021causal,
  title={Causal discovery in linear non-gaussian acyclic model with multiple latent confounders},
  author={Chen, Wei and Cai, Ruichu and Zhang, Kun and Hao, Zhifeng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={33},
  number={7},
  pages={2816--2827},
  year={2021},
  publisher={IEEE}
}
```

## Reminder and News
_February 19, 2024_

Maintenance and updates might not be timely since CADIMULC is personally developed without sponsors. 
Xuanzhi Chen is sorry about that, but opening the issue and advancing the community are always welcomed. 

Reach Out: <xuanzhichen.42@gmail.com>

## License
Copyright (C) 2022-2024 Xuanzhi Chen (DMIR lab, Guangdong University of Technology, China)

CADIMULC is downloaded for free, provided "as is" WITHOUT ANY EXPRESS OR IMPLIED WARRANTY;
CADIMULC is developed in hope of being beneficial for empirical data analysis in causation, but WITHOUT WARRANTY OF ABSOLUTELY ACCURATE INTERPRETATION.

<!--
## Acknowledgements
Xuanzhi Chen would like to thank the DMIR laboratory for offering him research opportunities, 
with special thanks to Ruichu Cai (蔡瑞初) of the lab director and Dongning Liu (刘冬宁) of the dean in School of Computer.

Jie Qiao (乔杰) and Zhiyi Huang (黄智毅) were willing to spend their time in personal discussions with Xuanzhi Chen 
about details in the paper;
Zhengming Chen (陈正铭) patiently helped point out initial mistakes in the paper;
thanks to other graduate students, such as Zeqin Yang (杨泽勤), Xiaokai Huang (黄晓楷), and Yu Xiang (向宇),
for their generosity of teaching when Xuanzhi chen was initially building the repository.

Finally,
Xuanzhi Chen owes a great debt to his advisor Wei Chen (陈薇) for her encouragement 
when Xuanzhi started studying causation two years ago
— "Do it, just have your own interest of research and your own rhythm of lifetime".
-->
