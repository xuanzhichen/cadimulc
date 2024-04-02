---

!!! success "Advice for developers if needed: Evaluation and visualization"
    Algorithms in CADIMULC simply represent the causation among variables, for both ground-truth 
    and learning results, as the directed pairs in an adjacency matrix with only two elements 0 and 1.
    
    If you incline to this representation of data structure in your work or research, 
    then `Evaluator` in CADIMULC might provide you convenience
    for evaluating the causal graph directly.

<h2 
style="font-size: x-large; font-weight: bold;"> 
<font color="IndianRed">Class:</font> <i>Evaluator</i>
</h2>

::: cadimulc.utils.evaluation.Evaluator
    handler: python
    options:
      members:
        - none
      show_root_heading: True
      show_source: false
      show_bases: false
      heading_level: 4

**---**

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>precision_pairwise</i> 
</h3>

::: cadimulc.utils.evaluation.Evaluator.precision_pairwise
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>recall_pairwise</i> 
</h3>

::: cadimulc.utils.evaluation.Evaluator.recall_pairwise
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>f1_score_pairwise</i> 
</h3>

::: cadimulc.utils.evaluation.Evaluator.f1_score_pairwise
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Primary Method:</font>  <i>evaluate_skeleton</i> 
</h3>

!!! note
    Construction of a network skeleton is the fundamental part relative to the procedure
    of hybrid-based approaches. 
    CADIMULC also provides simply way to evaluate the causal skeleton.
    Notice that performance of the hybrid-based approach largely depends on the initial
    performance of the causal skeleton learning.

::: cadimulc.utils.evaluation.Evaluator.evaluate_skeleton
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Secondary Method:</font>  <i>get_directed_pairs</i> 
</h3>

::: cadimulc.utils.evaluation.Evaluator.get_directed_pairs
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 5

<h3 
style="font-size: larger; font-weight: bold;"> 
<font color="IndianRed">Secondary Method:</font>  <i>get_pairwise_info</i> 
</h3>

::: cadimulc.utils.evaluation.Evaluator.get_pairwise_info
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