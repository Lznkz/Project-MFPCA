
# Table of Contents

1.  [Description](#org9a280f5)
2.  [Pipeline](#orgdeb34e4)
3.  [Methodology](#org05a12b4)
    1.  [1. Time Domain Registration [0, 1]](#org81afabb)
    2.  [2. B-Spline Smoothing (From Discrete to Functional Data)](#orge40a66e)
    3.  [3. Feature Extraction & Health-State Labeling (MFPCA, GMM, Youden Index)](#orgd240cca)
    4.  [4. Censored Test-Trajectory Smoothing & Candidate Matching (Adaptive Regression Spline)](#orgee1001d)
    5.  [5. Similarity-based Distance Calculation](#orge71edca)
    6.  [6. RUL Prediction & Interpretation](#org0162923)
4.  [Sensitivity Analysis](#orgdb8b1e2)
    1.  [Why using Adaptive Regresion Spline](#org65ee1e4)
    2.  [Distance Calculation has 2 formulas](#orgc1d5893)
5.  [Results](#org9b0e7a8)
6.  [Reference](#org972a50d)
    1.  [**Primary paper (reproduced in this repository):**](#orgfc21c43)
    2.  [**Methodology & Key Reference**](#org0623898)
    3.  [**Extras**](#org1097b11)

---


<a id="org9a280f5"></a>

# Description

This repository reproduces the multivariate functional data analysis (MFPCA)
pipeline from Yildirim, Franco & Lillo for predicting Remaining Useful Life
(RUL) of turbofan engines, applied to the NASA C-MAPSS FD001 dataset.

The pipeline combines B-spline smoothing with fleet-wide GCV-optimized
penalization, multivariate functional principal component analysis (Happ &
Greven, 2018) for health index construction, and a k-NN similarity-matching
scheme for RUL prediction on right-censored test trajectories.

This project was built independently as part of an undergraduate research
portfolio in Applied Mathematics, with a focus on faithful method reproduction
and transparent reporting of implementation decisions not fully specified in
the original paper.

**Result: RMSE ≈ 26.0 (mean-based prediction), within ~4% of the original paper&rsquo;s reported RMSE (~25).**

---


<a id="orgdeb34e4"></a>

# Pipeline

1.  [ Time Domain Registration [0, 1]​](#org81afabb)
2.  B-Spline Smoothing (From Discreet to Functional Data)
3.  Feature Extraction & Health-State Labeling (MFPCA, GMM, Youden Index)
4.  Censored Test-Trajectory Smoothing & Candidate Matching (Adaptive Regression Spline)
5.  Similarity-based Distance Calculation
6.  RUL Prediction & Interpretation

---


<a id="org05a12b4"></a>

# Methodology


<a id="org81afabb"></a>

## 1. Time Domain Registration [0, 1]

Since each engine in the C-MAPSS dataset operates under varying degrees of
initial wear and has a completely different lifespan (time to failure), the raw
multivariate data does not share a common temporal frame. This step therefore
rescales each engine&rsquo;s cycle index onto a common domain [0, 1], ensuring that
degradation curves can be directly compared stage-by-stage (e.g., early-life vs.
end-of-life) rather than cycle-by-cycle, without distorting the relative
progression of degradation within each unit.


<a id="orge40a66e"></a>

## 2. B-Spline Smoothing (From Discrete to Functional Data)

-   So we all know that raw data is often noisy and discrete, this also applies to
    CMAPSS which we gonna using. This step transforms these into continuous,
    smooth functional curves using cubic B-Spline with a second-derivative
    roughness penalty
-   In general, I will follow author&rsquo;s paper, which uses **Generalized
    Cross-Validation** (GCV) to determine the smoothing parameter (&ldquo;single
    $\lambda$ per sensor, optimized jointly across all training units&rdquo;), quadratic
    penalty on the second derivative, cubic spline, 20 knots => 24 basis was used
-   By following the formulation of the original paper:

$$GCV(\lambda) = \frac{1}{n}\sum_{i=1}^{n} GCV_i(\lambda)$$
$$GCV_i(\lambda) = \frac{(n_i+1)\, MSE_i(\lambda)}{[\,\text{trace}(I - H_i(\lambda))\,]^2}$$

where $n_i$ is the number of observed cycles for unit $i$, $MSE_i(\lambda)$ is
the residual mean squared error, and $H_i(\lambda)$ is the smoother hat matrix.
By applying **Generalized-Eigenvalue Parameterization** (Demmler-Reinsch) used to
evaluate $\text{trace}(H_i(\lambda))$ and $MSE_i(\lambda)$ efficiently boost
compute efficiency. Also those was implemented from scratch in Python (&rsquo;scipy&rsquo;,
&rsquo;scikit-fda&rsquo; basis objects), rather than relying on a higher-level
smoothing-spline library

---


<a id="orgd240cca"></a>

## 3. Feature Extraction & Health-State Labeling (MFPCA, GMM, Youden Index)

-   **Multivariate FPCA** (Happ & Greven, 2018) method extends the Karhunen–Loève
    Expansion, which was used to overcome the complexity of physical degradation
    dynamics in the CMAPSS dataset. As a result, **95.1%** of variance is explained by
    only the first principal component.
-   When examining the MFPC scores of the first PC in the training dataset, it can
    actually be modeled as a mixture of two normal distributions.  Using a
    Gaussian Mixture Model, we can split the training fleet into two health-state
    groups (&ldquo;low&rdquo;/&ldquo;big&rdquo; degradation rate).
-   From [this paper](#org1097b11), we can understand one reality: each engine operates with
    unknown, varying degrees of initial wear and manufacturing variation — not all
    engines are actually 100% identical. The first principal component scores
    could be related to that wear and manufacturing variation. Therefore, using
    the idea of the initial value of each sensor, we can label all units into two
    health-states. Using the **Youden Index**, we validate this classification with a
    mean Youden&rsquo;s J statistic of 0.90  averaged across all 9 sensors in training dataset, indicating
    strong discriminative power of the initial-value-based thresholds.
-   By applying the Youden Index cutoff, each unit in the testing dataset can be

categorized into one of two groups. Subsequently, candidate matches for each
test unit are identified through validation with the respective training subset.


<a id="orgee1001d"></a>

## 4. Censored Test-Trajectory Smoothing & Candidate Matching (Adaptive Regression Spline)

-   As we know, the units in the test dataset have varying lengths of recorded
    cycles, which makes it very challenging to actually reuse the smoothing
    parameter λ from the training dataset. (Also, in the paper, the author did not
    actually mention which method he used for this process.)
-   Therefore, instead of registering test units, both test units and training
    candidates (which are truncated to the same cycles as their test unit) are
    compared directly on the raw cycle domain [1, T<sub>test</sub>], where T<sub>test</sub> is the
    number of cycles observed so far for a given test unit. As a result, an
    **Adaptive Regression Spline** was used for each test unit and its candidates.
    
    > We also have one more condition to filter candidates, which keeps only training
    > engines that have outlived the relevant test engine. On the other hand,
    > training engines that fail earlier than the relevant test engine must be
    > eliminated. In this dataset, there is only one exception that didn&rsquo;t have any
    > candidates in its group; this may be caused by its initial value being very
    > close to the Youden Index cutoff point. Although the author did not mention a
    > solution for this case, my approach is to switch the candidate search to the
    > opposite health-state group.
-   With adaptive knot selection (1 knot per ~4 observations), we address the
    varying T<sub>test</sub> across units — a fixed basis size would otherwise cause
    underfitting on long trajectories or rank-deficiency on short ones. Also,
    **Cholesky Decomposition** was cached per unique T<sub>test</sub> value (not per unit) to
    avoid redundant computation across the many (test unit, candidate) pairs
    sharing the same length.


<a id="orge71edca"></a>

## 5. Similarity-based Distance Calculation

-   So the next step is calculate the distance between a test system’s sensor
    curve and training system curves in the same group. By using Euclidean
    ($L^2$) distance as it is the natural measure between two points in Euclidean
    space and represents the length of the line segment connecting two point
-   Before computing distance, each sensor&rsquo;s smoothed values are normalized by
    dividing by the sensor&rsquo;s mean value (computed across the full training set),
    following Eq. (20) of the original paper — this prevents sensors with larger
    raw magnitude (e.g., temperature, ~hundreds) from dominating the distance
    metric over sensors with smaller magnitude (e.g., pressure ratios)
    $$u_{ij}^*(t) = \frac{X_{ij}^{*}(t)}{\bar X_{j}^{*}(t)} \quad \text{i=1,....,n} \quad \text{j=1,....,J}$$
    where ${X_{ij}^{ *}(t)}$ is the sensor value of the ith system and the jth
    sensor at time t, ${X_{j}^{ *}(t)$ is the mean of the jth sensor, and J is the
    number of sensors
-   After that, the distance between the two engine curves in a multivariate sense is expressed as :
    $$d_{x,y}(t) = \sqrt{\sum_{j=1}^{J} (x_j(t) - y_j(t))^2}$$

&mdash;

-   Before computing distance, each sensor&rsquo;s evaluated values are normalized
    by dividing by $\bar X_j(t)$, the mean (or median) of that sensor&rsquo;s value
    across candidates at each of the 20 grid points, following the intent of
    Eq. (20) of the original paper — this prevents sensors with larger raw
    magnitude from dominating the distance metric over sensors with smaller
    magnitude.

> Several time-varying normalization schemes were tested (mean vs. median as
> the center, with and without an additional per-timepoint standard
> deviation scaling). All variants performed similarly or worse than the
> simplest static-mean normalization; standard-deviation scaling in
> particular tended to flatten meaningful variation across candidates and
> degraded RMSE. The original static-mean formulation was therefore kept.

-   Since candidate trajectories have varying lengths (T<sub>test</sub>), each fitted
    spline is evaluated at a fixed grid of 20 points spanning [1, T<sub>test</sub>],
    rather than at the full set of raw observation cycles. This provides a
    common, fixed-length representation for every (test unit, candidate)
    pair, which is required before a pointwise Euclidean distance can be
    computed — comparing curves of different raw lengths directly would
    otherwise be undefined.
-   Two formulations of multivariate Euclidean distance were then compared:
    -   **Joint**: $D_{joint} = \sqrt{\sum_{j=1}^{9}\sum_{t=1}^{20} (x_j(t) -
            y_j(t))^2}$ — flattens all 9 sensors and 20 time points into a single
        vector before taking one square root. This matches Eq. (21) as written
        in the paper&rsquo;s text.
    -   **Per-sensor sum**: $D_{sum} = \sum_{j=1}^{9} \sqrt{\sum_{t=1}^{20}
            (x_j(t) - y_j(t))^2}$ — computes the Euclidean distance for each
        sensor independently, then sums across sensors. Cross-referencing the
        author&rsquo;s official FLARE implementation on GitHub shows this is the
        formula actually used in code, despite Eq. (21) in the text describing
        the joint form.

> Empirically, per-sensor sum reduced RMSE from ~30 to 25.99, matching the
> author&rsquo;s actual implementation rather than the text of Eq. (21).
> Investigating why: post-normalization variance varies roughly 60-fold
> across the 9 sensors (from 1.1e-7 for T24 to 7.0e-6 for T50). Under the
> joint formula, this imbalance lets 2-3 high-variance sensors dominate the
> squared sum, effectively discarding information from the remaining
> sensors. The per-sensor-summed formula avoids this by weighting each
> sensor&rsquo;s contribution additively rather than quadratically-jointly,
> explaining both its empirical advantage and its agreement with the
> author&rsquo;s code.

&mdash;


<a id="org0162923"></a>

## 6. RUL Prediction & Interpretation

-   After we got the distance of each candidates with the corresponding test engine, we can find the k number of units which has min
-   After that we can add thing related to first and second derivatives


<a id="orgdb8b1e2"></a>

# Sensitivity Analysis


<a id="org65ee1e4"></a>

## [Why using Adaptive Regresion Spline](#orgee1001d)

-   As we know, the units in the test dataset have varying lengths of recorded
    cycles, which makes it very challenging to actually reuse the smoothing
    parameter λ from the training dataset. (Also, in the paper, the author did not
    actually mention which method he used for this process.)


<a id="orgc1d5893"></a>

## [Distance Calculation has 2 formulas](#orge71edca)

The per-sensor formulation likely outperforms the joint formula (Eq. 21)
because the joint L2-norm allows any single sensor with disproportionately
large post-normalization variance to dominate the total distance, while
the per-sensor-summed formulation bounds each sensor&rsquo;s contribution to a
single additive term. This suggests the normalization scheme (Eq. 20,
dividing by a static per-sensor mean) does not fully equalize variance
across the 9 sensors in this dataset.
\### Future Work
A Kneedle-based elbow detection on MFPC1-vs-cycle trajectories was
considered as a way to segment normalization into pre-/post-degradation-
onset phases. Given that all tested time-varying normalization schemes
(median-based, std-scaled) underperformed the simpler static-mean
normalization in this study, this direction was not pursued further, but
may be worth revisiting with test units having longer observed trajectories.


<a id="org9b0e7a8"></a>

# Results

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Prediction method</th>
<th scope="col" class="org-right">RMSE</th>
<th scope="col" class="org-right">Correct Pred.</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">Original paper (FLARE - Mean)</td>
<td class="org-right">25.41</td>
<td class="org-right">7</td>
</tr>

<tr>
<td class="org-left">Original paper (FLARE - Median)</td>
<td class="org-right">25.74</td>
<td class="org-right">1</td>
</tr>

<tr>
<td class="org-left">This work - Proposed Method (Mean)</td>
<td class="org-right">25.98</td>
<td class="org-right">2</td>
</tr>

<tr>
<td class="org-left">This work - Propose Method (Median)</td>
<td class="org-right">27.73</td>
<td class="org-right">1</td>
</tr>
</tbody>
</table>

---


<a id="org972a50d"></a>

# Reference


<a id="orgfc21c43"></a>

## **Primary paper (reproduced in this repository):**

> Yildirim, C., Lillo, R. E., & Franco-Pereira, A. M. (2025). Health Prognostics in Multi-Sensor Systems Based on Multivariate Functional Data Analysis. Available at SSRN 4907886.


<a id="org0623898"></a>

## **Methodology & Key Reference**

> Happ, C., & Greven, S. (2018). Multivariate functional principal component analysis for data observed on different (dimensional) domains. Journal of the American Statistical Association, 113(522), 649-659.


<a id="org1097b11"></a>

## **Extras**

> Saxena, Abhinav, et al. &ldquo;Damage propagation modeling for aircraft engine run-to-failure simulation.&rdquo; 2008 international conference on prognostics and health management. IEEE, 2008.

