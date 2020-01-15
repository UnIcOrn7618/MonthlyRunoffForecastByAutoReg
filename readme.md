# Code repository for ["Two-stage Variational Mode Decomposition and Support Vector Regression"](https://www.hydrol-earth-syst-sci-discuss.net/hess-2019-565/#discussion)

This study heavily relies on the open-source software. Pandas (McKinney, 2010) and numpy (Stéfan et al., 2011) were used to manage and process streamflow data. Matlab was used to perform variational mode decomposition ([VMD](https://ieeexplore.ieee.org/document/6655981)), ensemble empirical mode decomposition ([EEMD](https://doi.org/10.1142/S1793536909000047)), and discrete wavelet transform ([DWT](https://www.mathworks.com/help/wavelet/ref/dwt.html)) of streamflow and compute the partial autocorrelation coefficient (PACF) of subsignals. The matlab implementations of VMD and EEMD come from Dragomiretskiy and Zosso (2014) and Wu and Huang (2009), respectively. The DWT was performed based on matlab build-in toolbox (“Wavelet 1-D” in “Wavelet Analyzer”). The SSA was performed based on [a python program developed by  Jordan D'Arcy](https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition). The SVR model in [scikit-learn](https://scikit-learn.org/stable/) (Pedregosa et al., 2011) was used to train SVR models. [scikit-optimize](https://scikit-optimize.github.io/) (Tim et al., 2018) was used to tune the SVR models. Matplotlib (Hunter, 2007) was used to draw figures.

## How to validate the research results

1. Clone this repository from Github. Run the following code in CMD or git-bash.

    ```
    git clone https://github.com/zjy8006/MonthlyRunoffForecastByAutoReg
    ```

2. Open MATLAB for decomposing monthly runoff using EEMD, VMD, DWT and MODWT, and building ARIMA model. Go to the root directory. 

    ```
    % Run the following code in command window.
    >> cd where_you_save:/MonthlyRunoffForecastByAutoReg
    ```

3. Open this repository with [vscode](https://code.visualstudio.com/) for other tasks. You can run code with [code runner](https://marketplace.visualstudio.com/items?itemName=formulahendry.code-runner) extension.

## Shift-invariant test and  boundary effect analysis of VMD

* Run **"tools/vmd_shift_invariant_test.m"**

## Monthly runoff decomposition

* Run **"/tools/RUN_EEMD.mlx"** for EEMD of monthly runoff.
* Run **"/tools/RUN_VMD.mlx"** for VMD of monthly runoff.
* Run **"/tools/RUN_DWT.mlx"** for DWT of monthly runoff.
* Run **"/tools/RUN_MODWT.mlx"** for MODWT of monthly runoff.
* Run **"/tools/ssa_decompose.py"** for SSA of monthly runoff.

## Determine the input predictors

* Run **"/tools/compute_pacf.m"** for computing PACF.
* Run **"/\*\*/projects/variables.py"** for selecting input predictors.
predictors.

## Generate training, development and testing samples

* Run **"/\*\*/projects/generate_samples.py"**

## Support Vector Regression

The 'SVR' in [scikit-learn](https://scikit-learn.org/stable/) was used to build support vector regression (SVR) models. The 'gp_minimize' (Bayesian optimization based on Gaussian process) in [scikit-optimize](https://scikit-optimize.github.io/) was used to optimize SVR models. The SVR model optimized by Bayesian optimization are organized in './tools/models.py'.


## Reference

* McKinney, W., 2010. Data Structures for Statistical Computing in Python, pp. 51–56.
* Stéfan, v.d.W., Colbert, S.C., Varoquaux, G., 2011. The NumPy Array: A Structure for Efficient Numerical Computation. A Structure for Efficient Numerical Computation. Comput. Sci. Eng. 13 (2), 22–30.
* Dragomiretskiy, K., Zosso, D., 2014. Variational Mode Decomposition. IEEE Trans. Signal Process. 62 (3), 531–544.
* Wu, Z., Huang, N.E., 2009. Ensemble Empirical Mode Decomposition: a Noise-Assisted Data Analysis Method. Adv. Adapt. Data Anal. 01 (01), 1–41.
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, É., 2011. Scikit-learn. Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830.
* Tim, H., MechCoder, Gilles, L., Iaroslav, S., fcharras, Zé Vinícius, cmmalone, Christopher, S., nel215, Nuno, C., Todd, Y., Stefano, C., Thomas, F., rene-rex, Kejia, (K.) S., Justus, S., carlosdanielcsantos, Hvass-Labs, Mikhail, P., SoManyUsernamesTaken, Fred, C., Loïc, E., Lilian, B., Mehdi, C., Karlson, P., Fabian, L., Christophe, C., Anna, G., Andreas, M., and Alexander, F.: Scikit-Optimize/Scikit-Optimize: V0.5.2, Zenodo, 2018.
* Hunter, J.D., 2007. Matplotlib. A 2D Graphics Environment. Computing in Science & Engineering 9, 90–95.
