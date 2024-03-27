def PCMCI():
    """
    PCMCI causal discovery for time series datasets.

    PCMCI is a causal discovery framework for large-scale time series
    datasets. This class contains several methods. The standard PCMCI method
    addresses time-lagged causal discovery and is described in Ref [1] where
    also further sub-variants are discussed. Lagged as well as contemporaneous
    causal discovery is addressed with PCMCIplus and described in [5]. See the
    tutorials for guidance in applying these methods.

    PCMCI has:

    * different conditional independence tests adapted to linear or
      nonlinear dependencies, and continuously-valued or discrete data (
      implemented in ``tigramite.independence_tests``)
    * (mostly) hyperparameter optimization
    * easy parallelization (separate script)
    * handling of masked time series data
    * false discovery control and confidence interval estimation


    Notes
    -----

    .. image:: mci_schematic.*
       :width: 200pt

    In the PCMCI framework, the dependency structure of a set of time series
    variables is represented in a *time series graph* as shown in the Figure.
    The nodes of a time series graph are defined as the variables at
    different times and a link indicates a conditional dependency that can be
    interpreted as a causal dependency under certain assumptions (see paper).
    Assuming stationarity, the links are repeated in time. The parents
    :math:`\mathcal{P}` of a variable are defined as the set of all nodes
    with a link towards it (blue and red boxes in Figure).

    The different PCMCI methods estimate causal links by iterative
    conditional independence testing. PCMCI can be flexibly combined with
    any kind of conditional independence test statistic adapted to the kind
    of data (continuous or discrete) and its assumed dependency types.
    These are available in ``tigramite.independence_tests``.

    NOTE: MCI test statistic values define a particular measure of causal
    strength depending on the test statistic used. For example, ParCorr()
    results in normalized values between -1 and 1. However, if you are
    interested in quantifying causal effects, i.e., the effect of
    hypothetical interventions, you may better look at the causal effect
    estimation functionality of Tigramite.

    References
    ----------

    [1] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic,
           Detecting and quantifying causal associations in large nonlinear time
           series datasets. Sci. Adv. 5, eaau4996 (2019)
           https://advances.sciencemag.org/content/5/11/eaau4996

    [5] J. Runge,
           Discovering contemporaneous and lagged causal relations in
           autocorrelated nonlinear time series datasets
           http://www.auai.org/~w-auai/uai2020/proceedings/579_main_paper.pdf

    Parameters
    ----------
    dataframe : data object
        This is the Tigramite dataframe object. Among others, it has the
        attributes dataframe.values yielding a numpy array of shape (
        observations T, variables N) and optionally a mask of the same shape.
    cond_ind_test : conditional independence test object
        This can be ParCorr or other classes from
        ``tigramite.independence_tests`` or an external test passed as a
        callable. This test can be based on the class
        tigramite.independence_tests.CondIndTest.
    verbosity : int, optional (default: 0)
        Verbose levels 0, 1, ...

    Attributes
    ----------
    all_parents : dictionary
        Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
        the conditioning-parents estimated with PC algorithm.
    val_min : dictionary
        Dictionary of form val_min[j][(i, -tau)] = float
        containing the minimum absolute test statistic value for each link estimated in
        the PC algorithm.
    pval_max : dictionary
        Dictionary of form pval_max[j][(i, -tau)] = float containing the maximum
        p-value for each link estimated in the PC algorithm.
    iterations : dictionary
        Dictionary containing further information on algorithm steps.
    N : int
        Number of variables.
    T : dict
        Time series sample length of dataset(s).
    """
    pass