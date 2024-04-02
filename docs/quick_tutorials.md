# Quick Tutorials
Here is an exceptionally simple example to help take a look on the "micro" workflow of the 
causal discovery task. By attaching to this mini-causal-discovery procedure, 
you could lean to experience a complete causal discovery in minutes.

!!! note
    This page is currently under building.

[//]: # (## 1. Installation)

[//]: # (To install CADIMULC, run the following command from your Python virtual environment:)

[//]: # (```commandline)

[//]: # (pip install cadimulc)

[//]: # (```)

[//]: # ()
[//]: # (## 2. SCMs Data Generation)

[//]: # (In order to discover a causal graph from data, we might first learn about how to )

[//]: # (generate the data based on a "graphical" causal model.)

[//]: # (```python)

[//]: # (from cadimulc.utils.generation import Generator)

[//]: # (```)

[//]: # (```python linenums="1")

[//]: # (np.random.seed&#40;42&#41;)

[//]: # (random.seed&#40;42&#41;)

[//]: # ()
[//]: # (generator = Generator&#40;)

[//]: # (        graph_node_num=5,)

[//]: # (        sample=5000,)

[//]: # (        causal_model='hybrid_nonlinear',)

[//]: # (        noise_type='Gaussian')

[//]: # (    &#41;)

[//]: # ()
[//]: # (ground_truth, dataset = generator.run_generation_procedure&#40;&#41;.unpack&#40;&#41;)

[//]: # (```)

[//]: # (The causal graph we wish to discover serves as a representation for its deeper)

[//]: # (**structure causal models** &#40;SCMs&#41;, which reflects our **priori causal assumption** towards to)

[//]: # (the data generation. )

[//]: # (See [SCMs Data Generation]&#40;url&#41; for a brief introduction.)

[//]: # ()
[//]: # (!!! note "Reminder")

[//]: # (    Step-2 is not a must, we can always apply the algorithm)

[//]: # (    in step-3 for other established dataset. )

[//]: # (    The SCMs assumption, however,  does not necessarily hold over the dataset )

[//]: # (    we want to analyze.)

[//]: # (    Keeping this in mind might be helpful for us to objectively interpret the )

[//]: # (    hypothetical causation learned from the empirical data.)

[//]: # ()
[//]: # (## 3. Hybrid-Based Causal Discovery)

[//]: # (### 3.1 Without Latent Confounders)

[//]: # (Applying hybrid-based approaches is super easy. Take general non-linear causal discovery.)

[//]: # (```python)

[//]: # (from cadimulc.hybrid_algorithms import NonlinearMLC)

[//]: # (```)

[//]: # (Two of the primary parameter setup, for hybrid-based methodology,)

[//]: # (are respectively the conditional-independence-test &#40;CIT&#41;)

[//]: # (approach,)

[//]: # (```python)

[//]: # (ind_test = 'kernel_ci')

[//]: # (```)

[//]: # (and the functional-causal-models &#40;FCMs&#41; regressor.)

[//]: # (```python)

[//]: # (from pygam import LinearGAM)

[//]: # (```)

[//]: # (```python )

[//]: # (nonlinear_mlc = NonlinearMLC&#40;)

[//]: # (    regressor=LinearGAM&#40;&#41;,  )

[//]: # (    ind_test=ind_test)

[//]: # (&#41;)

[//]: # (```)

[//]: # (Flexible adjustments of the empirical &#40;non-linear&#41; regressor are available.)

[//]: # ()
[//]: # (For example, if you don't mind too much computational consumption, normally, choose a Neural Network)

[//]: # (as the regression model would obtain slightly better performance.)

[//]: # (```python )

[//]: # (from sklearn.neural_network import MLPRegressor)

[//]: # ()
[//]: # (nonlinear_mlc = NonlinearMLC&#40;)

[//]: # (    regressor=MLPRegressor&#40;&#41;,  )

[//]: # (    ind_test=ind_test)

[//]: # (&#41;)

[//]: # (```)

[//]: # (!!! warning)

[//]: # (    Theoretically, )

[//]: # (Now, conduct non-linear causal discovery that is adaptive to multiple latent confounders.)

[//]: # (```python )

[//]: # (nonlinear_mlc.fit&#40;dataset=dataset&#41;)

[//]: # (```)

[//]: # (The truth is same if you would like to presume linearity relations entailed by the dataset.)

[//]: # (Simply replace `NonlinearMLC` as `MLC-LiNGAM`, and repeat the same steps.)

[//]: # (```python)

[//]: # (from cadimulc.hybrid_algorithms import MLCLiNGAM)

[//]: # (```)

[//]: # (Notice that parameter combination for linear causal discovery ...)

[//]: # (```python)

[//]: # (from sklearn.linear_model import LinearRegression)

[//]: # (ind_test = '')

[//]: # ()
[//]: # (mlc_lingam = MLCLiNGAM&#40;)

[//]: # (    regressor=LinearRegression&#40;&#41;,  )

[//]: # (    ind_test=ind_test)

[//]: # (&#41;)

[//]: # (```)

[//]: # (The `fit` procedure involves a two-steps hybrid causal discovery strategy. )

[//]: # (For technical and theoretical details about the approaches, )

[//]: # (please refer to [Hybrid-Based Approaches]&#40;url&#41;.   )

[//]: # ()
[//]: # (### 3.2 With Latent Confounders)

[//]: # ()
[//]: # (## 4. Evaluation and Visualization)

[//]: # (For example, )

[//]: # (simply input two of the adjacency matrices representing the ground-truth and the learning result)

[//]: # (respectively, `Evaluator` will calculate the metrics relative to causal graphs,)

[//]: # (akin to the common indicators used in pattern recognition, such as )

[//]: # (**causal edge precision**, **causal edge recall**, )

[//]: # (and **causal edge f1-score**.)

[//]: # ()
[//]: # (## 5. Summary)

[//]: # (online-learning or reinforced learning.)