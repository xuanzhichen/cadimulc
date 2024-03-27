# Evaluation

---

<h2 style="font-size: x-large; font-weight: bold;"> class: Evaluator </h2> 

::: cadimulc.utils.evaluation.Evaluator
    handler: python
    options:
      members:
        - none
      show_root_heading: false
      show_source: false
      show_bases: false
      heading_level: 2

!!! example

    ``` py 
    def bubble_sort(items):
        for i in range(len(items)):
            for j in range(len(items) - 1 - i):
                if items[j] > items[j + 1]:
                    items[j], items[j + 1] = items[j + 1], items[j]
    ```
---

<h3 style="font-size: larger; font-weight: bold;"> method: precision_pairwise </h3>

::: cadimulc.utils.evaluation.Evaluator.precision_pairwise
    handler: python
    options:
      show_root_heading: false
      show_source: true
      heading_level: 3

!!! tip
    $$
    \operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
    $$