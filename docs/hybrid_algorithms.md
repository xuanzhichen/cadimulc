
---

<h2 
style="font-size: x-large; font-weight: bold;">
<font color="IndianRed">Class:</font> <i>MLC-LiNGAM</i> <sup>[1]</sup> 
</h2> 

::: cadimulc.hybrid_algorithms.hybrid_algorithms.MLCLiNGAM
    handler: python
    options:
      members:
        - __init__
      show_root_heading: True
      show_source: false
      show_bases: true
      heading_level: 4

!!! warning "The output corresponding to the MLC-LiNGAM algorithm"
    In the field of causal discovery,
    causal graphs are usually represented as the directed acyclic
    graph (DAG) or the causal order. 
    The existence of latent confounders, however, might result
    in the causal relations that cannot be determined by algorithms. 

    Correspondingly, the estimated causal graph,
    by the MLC-LiNGAM or Nonlinear-MLC algorithm in CADIMULC,
    is represented as the partial directed acyclic graph or 
    the partial causal order.

---

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>fit</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.MLCLiNGAM.fit
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Private Method:</font>  <i>_stage_1_learning</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.MLCLiNGAM._stage_1_learning
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Private Method:</font>  <i>_stage_2_learning</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.MLCLiNGAM._stage_2_learning
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Private Method:</font>  <i>_stage_3_learning</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.MLCLiNGAM._stage_3_learning
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h2 
style="font-size: x-large; font-weight: bold;">
<font color="IndianRed">Class:</font> <i>Nonlinear-MLC</i> <sup>[2]</sup> 
</h2> 

::: cadimulc.hybrid_algorithms.hybrid_algorithms.NonlinearMLC
    handler: python
    options:
      members:
        - __init__
      show_root_heading: True
      show_source: false
      show_bases: true
      heading_level: 4

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>fit</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.NonlinearMLC.fit
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Private Method:</font>  <i>_clique_based_causal_inference</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.NonlinearMLC._clique_based_causal_inference
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<!--
::: cadimulc.utils.hybrid_algorithms.hybrid_algorithms.NonlinearMLC.\_\_init__
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 3
-->

<h2 
style="font-size: x-large; font-weight: bold;">
<font color="IndianRed">Auxiliary Class:</font> <i>GraphPatternManager</i> 
</h2> 

::: cadimulc.hybrid_algorithms.hybrid_algorithms.GraphPatternManager
    handler: python
    options:
      members:
        - none
      show_root_heading: True
      show_source: false
      show_bases: true
      heading_level: 4

---

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>recognize_maximal_cliques_pattern</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.GraphPatternManager.recognize_maximal_cliques_pattern
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>find_adjacent_set</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_algorithms.GraphPatternManager.find_adjacent_set
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h2 
style="font-size: x-large; font-weight: bold;"> 
<font color="IndianRed">Base Class:</font> <i>HybridFrameworkBase</i>
</h2> 

::: cadimulc.hybrid_algorithms.hybrid_framework.HybridFrameworkBase
    handler: python
    options:
      members:
        - __init__
      show_root_heading: True
      show_source: false
      show_bases: false
      heading_level: 4

--- 

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font> <i>_causal_skeleton_learning</i> 
</h3>

::: cadimulc.hybrid_algorithms.hybrid_framework.HybridFrameworkBase._causal_skeleton_learning
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

## Reference
[1] Chen, Wei, Ruichu Cai, Kun Zhang, and Zhifeng Hao.
"[Causal discovery in linear non-gaussian acyclic model with multiple latent confounders. ](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Causal+discovery+in+linear+non-gaussian+acyclic+model+with+multiple+latent+confounders&btnG=#d=gs_cit&t=1711554753714&u=%2Fscholar%3Fq%3Dinfo%3AzEuwtDsRA24J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den)"
*IEEE Transactions on Neural Networks and Learning Systems.* 2021.

[2] Chen, Xuanzhi, Wei Chen, Ruichu Cai. 
"[Non-linear Causal Discovery for Additive Noise Model with
    Multiple Latent Confounders](https://xuanzhichen.github.io/work/papers/nonlinear_mlc.pdf)". *Xuanzhi's Personal Website.* 2023.

[3] Shimizu, Shohei, Patrik O. Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan. 
"[A linear non-Gaussian acyclic model for causal discovery.](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=OpLI4xcAAAAJ&citation_for_view=OpLI4xcAAAAJ:7PzlFSSx8tAC)" 
*Journal of Machine Learning Research.* 2006.

[4] Hoyer, Patrik, Dominik Janzing, Joris M. Mooij, Jonas Peters, and Bernhard Schölkopf.
"[Nonlinear causal discovery with additive noise models.](https://proceedings.neurips.cc/paper/2008/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html)" 
*Advances in neural information processing systems*. 2008.

[5] Spirtes, Peter, Clark N. Glymour, and Richard Scheines.
Causation, prediction, and search.
MIT press, 2000.

