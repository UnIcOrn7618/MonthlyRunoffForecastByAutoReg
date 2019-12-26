#' Thanks John Quilty for providing this R script

#' Example R script to show the shift-invariancy and sensitivity to adding additional data points
#' when using VMD for to proivde inputs for operational forecasting tasks.
#' Note: run this script line by line instead of executing it in its entirety (as only the last plot
#' will be displayed in the latter case).
set.seed(123)
install.packages("EMD",repos = "http://cran.us.r-project.org")
install.packages("vmd",repos = "http://cran.us.r-project.org")
library(EMD) # this is so that we can use the sunspot series as a test case
library(vmd)
data("sunspot")
x <- sunspot$sunspot
###############################################################################################
#' CHECK 1: Is VMD shift-invariant?
#' If yes, any shifted copy of an IMF from a VMD decomposition, similar to a shifted copy of the
#' original time series, should be maintained.
#'
#' For example, given the sunspot time series x (of length 386) we can generate a 1-step
#' advanced copy of the original time series as follows:
#'
#' x0 = x[1:385]
#' x1 = x[2:386] # this is a 1-step advanced version of x0
#'
#' Obviously, shift-invariancy is preserved between x0 and x1 since x0[2:385] = x1[1:384]
#'
#'
#' For shift-invariancy to be preserved for VMD, we should observe, for example, that the
#' VMD IMF1 components for x0 (x0.IMF1) and x1 (x1.IMF1) should be exact copies of one another,
#' advanced by a single step.
#'
#' I.e., x0.IMF1[2:385] should equal x1.IMF1[1:384] if shift-invariancy is
#' preserved. However, this is not the case for VMD as shown below. Interestingly, we see
#' a high level of error at the boundaries of the time series, of high importance in
#' operational forecasting tasks.
###############################################################################################
# lag/advance sunspot data by 1-step
x0 = x[1:385]
x1 = x[2:386]
# perform VMD on both original and advanced time series
x0.vmd = vmd(x0)$getResult()$u # default settings
x1.vmd = vmd(x1)$getResult()$u # default settings
# plot IMF1 for x0.vmd and x1.vmd
plot(x0.vmd[2:385,1],type='l')
lines(x1.vmd[1:384,1],col='red')
# plot error (which should be a straight line of 0s if shift-invariancy was preserved)
err = x0.vmd[2:385] - x1.vmd[1:384]
plot(err)
# check the level of error (as measured by the mean square error) between the IMF components
mse = mean(err^2)
###############################################################################################
#' CHECK 2: The impact of appending data points to a time series then performing VMD, analogous
#' to the case in operational forecasting when new data becomes available and an updated forecast
#' is made using the newly arrived data.
#'
#' Ideally, for forecasting situations, when new data is appended to a time series an some pre-
#' processing is performed, it should not have an impact on previous measurements of the pre-
#' processed time series.
#'
#' For example, if IMF1_1:N represents the IMF1, which has N total measurements and was derived
#' by applying VMD to x_1:N then we would expect that when we perform VMD when x is appended with
#' another measurement, i.e., x_1:N+1, resulting in IMF1_1:N+1 that the first 1:N measurements in
#' IMF1_1:N+1 are equal to IMF1_1:N. In other words, IMF_1:N+1[1:N] = IMF1_1:N[1:N].
#'
#' Again, we see that this is not the case. Appending an additional observation to the time series
#' results in the updated VMD components to be entirely different then the original (as of yet
#' updated) VMD components. Interestingly, we see a high level of error at the boundaries of the
#' time series, of high importance in operational forecasting tasks.
###############################################################################################
# extend x with an additional measurement
x_1_385 = x[1:385]
x_1_386 = x[1:386]
# perform VMD on original and extended time series
x_1_385.vmd = vmd(x_1_385)$getResult()$u # default settings
x_1_386.vmd = vmd(x_1_386)$getResult()$u # default settings
# plot IMF1 for x_1_385.vmd and x_1_386.vmd
plot(1:386, x_1_386.vmd[1:386,1],type='l')
lines(1:385, x_1_385.vmd[1:385,1],col='red')
#' plot error (which should be a straight line of 0s if appending an additional observation has
#' no impact on VMD)
err.append = x_1_385.vmd[1:385] - x_1_386.vmd[1:385]
plot(err.append)
# check the level of error (as measured by the mean square error) between the IMF components
mse.append = mean(err.append^2)
# the problem gets exasperated if it is not a single time point that is updated, but several
x_1_300 = x[1:300]
# perform VMD on original and extended time series
x_1_300.vmd = vmd(x_1_300)$getResult()$u # default settings
# plot IMF1 for x_1_385.vmd and x_1_386.vmd
plot(1:386, x_1_386.vmd[1:386,1],type='l')
lines(1:300, x_1_300.vmd[1:300,1],col='red')
#' plot error (which should be a straight line of 0s if appending an additional observation has
#' no impact on VMD)
err.append.ext = x_1_300.vmd[1:300] - x_1_386.vmd[1:300]
plot(err.append.ext)
# check the level of error (as measured by the mean square error) between the IMF components
mse.append.ext = mean(err.append.ext^2)
# EOF