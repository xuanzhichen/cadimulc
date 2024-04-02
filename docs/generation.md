---
!!! success "Advice for developers if needed: SCMs data generation"
    `Generator` in CADIMULC serves as a framework of general data generation in 
    the task of causal discovery. 
    Default settings of hyperparameters (e.g. parameters of specific causal function) 
    in the `Generator` might require being fine-tuned depends on different purposes
    for simulation.

    Users could develop their own "causal simulator" in data analysis based on their need 
    interest, by following the data generation template in `Generator`.

<h2 
style="font-size: x-large; font-weight: bold;"> 
<font color="IndianRed">Class:</font> <i>Generator</i>
</h2>

::: cadimulc.utils.generation.Generator
    handler: python
    options:
      members:
        - __init__
      show_root_heading: True
      show_source: false
      show_bases: flase
      heading_level: 4

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>run_generation_procedure</i> 
</h3>

::: cadimulc.utils.generation.Generator.run_generation_procedure
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Private Method:</font>  <i>_generate_dag</i> 
</h3>

::: cadimulc.utils.generation.Generator._generate_dag
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Private Method:</font>  <i>_generate_data</i> 
</h3>

::: cadimulc.utils.generation.Generator._generate_data
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

!!! example "examples for these methods"
    CADIMULC is a light Python repository without sophisticated library API design.
    Documentation on this page is meant to provide introductory materials of the practical tool
    as to causal discovery.
    For running example, 
    please simply check out [Quick Tutorials](https://xuanzhichen.github.io/cadimulc/) for the straightforward usage in the "micro" workflow of 
    causal discovery.

## Reference
[1] Shimizu, Shohei, Patrik O. Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan. 
"[A linear non-Gaussian acyclic model for causal discovery.](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=OpLI4xcAAAAJ&citation_for_view=OpLI4xcAAAAJ:7PzlFSSx8tAC)" 
*Journal of Machine Learning Research.* 2006.

[2] Chen, Wei, Ruichu Cai, Kun Zhang, and Zhifeng Hao.
"[Causal discovery in linear non-gaussian acyclic model with multiple latent confounders. ](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Causal+discovery+in+linear+non-gaussian+acyclic+model+with+multiple+latent+confounders&btnG=#d=gs_cit&t=1711554753714&u=%2Fscholar%3Fq%3Dinfo%3AzEuwtDsRA24J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)"
*IEEE Transactions on Neural Networks and Learning Systems.* 2021.

[3] Bühlmann, Peter, Jonas Peters, and Jan Ernest. 
"[CAM: Causal additive models, high-dimensional order search and penalized regression.](https://scholar.google.com/schoxlar?hl=en&as_sdt=0%2C5&q=causal+additive+models+with+unobserved+variables&oq=causal+additive)" 
2014. 

[4] Maeda, Takashi Nicholas, and Shohei Shimizu. 
"[Causal additive models with unobserved variables.](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=causal+additive+models+with+unobserved+variables&oq=causal+additive)" 
*In Uncertainty in Artificial Intelligence.* 2021.
