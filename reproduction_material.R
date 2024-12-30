################################################################################
#                     Reproduction material for the paper:
#
#     An Alternating Direction Method of Multipliers Algorithm for the 
#                Weighted Fused LASSO  Signal Approximator
#
################################################################################

# Load required libraries
library(flsa)
library(wflsa)
library(dplyr)
library(microbenchmark)
library(progress)
library(magick)
library(narray)
library(wflsa)
library(foreach)
library(doParallel)
library(ggplot2)
library(grid)
library(gridExtra)

# Load utility functions
source("utils.R")

################################################################################
#                       Section ???: Runtime comparison
################################################################################

# Random binary graph ----------------------------------------------------------
# Set seed for reproducibility
set.seed(42)

# Constants
lambda1 <- 1
lambda2 <- 1
k_maxGrpNum <- 8
n_repls <- 2

#' This tibble contains all the parameter settings. It includes combinations of 
#' 'p' (size of the square weight matrix) and 'density' (density of the binary values).
parameter_settings <- tibble(expand.grid(
  p = c(10, 50, seq(100,1000,by=100)), 
  density = c(.5)#seq(.1, 1, by = .1)
))

# number of parameter settings
n_parameter_settings <- nrow(parameter_settings)

# Create a progress bar
pb <- progress::progress_bar$new(total = n_parameter_settings)

#' Go over each of the parameter settings and check the run time
res <- lapply(1:n_parameter_settings, function(i) {
  
  # Get the parameters
  p <- as.integer(parameter_settings[i, 'p'])
  density <- as.double(parameter_settings[i,'density'])
  
  # Create the random binary matrix weight matrix
  W <- random_binary_weight_matrix(p, density)
  
  # Create the corresponding connListObj needed by the flsa package
  connListObj <- create_connListObj(W)
  
  # generate the raw data 
  y <- rnorm(p)
  
  # Applies the flsa function to the data
  FLSA <- function() {
    # if there are no connections what so ever
    if (connListObj$connList_is_null) {
      # lambda2 is zero, since there is no smoothness penalty
      c(flsa::flsa(y, lambda1 = lambda1, lambda2 = 0, thr = 1e-7))
    } else {
      flsa_obj <- flsa::flsa(y, lambda1 = lambda1, connListObj = connListObj$connList, 
                             thr = 1e-7, maxGrpNum = k_maxGrpNum * length(y))
      c(flsaGetSolution(flsa_obj, lambda1 = lambda1, lambda2 = lambda2))
    }
  }
  
  # Applies the wflsa package to the data
  wFLSA <- function() {
    wflsa::wflsa(y, W, lambda1 = lambda1, lambda2 = lambda2, eps = 1e-7)$betas[[1]]
  }
  
  # Assesses the runtime 
  time <- microbenchmark::microbenchmark(wFLSA(), 
                                         FLSA(), times = n_repls)
  
  # Get the difference
  diff <- replicate(n_repls, mean((wFLSA() - FLSA())^2))
  pb$tick()
  
  return(list(time = time, difference = diff))
})

# Create plot
p_random_graph <- create_plot(lapply(res, function(x) x$time), 
                              parameter_settings = parameter_settings, 
                              title = "Random Graph")

p_random_graph_diff <- ggplot2::ggplot(data = data.frame(
  p = base::rep(parameter_settings$p, each = n_repls),
  difference = unlist(lapply(res, function(x) x$difference))
), aes(x = p, y = difference)) + 
  geom_point() + 
  geom_smooth() + 
  ylab("Mean Squared Difference") + 
  xlab("p") + 
  ggtitle("Mean Squared Difference between wFLSA and FLSA") + 
  theme_minimal()

# Classic 1-D FLSA -------------------------------------------------------------
# Set seed for reproducibility
set.seed(42)

# Constants
lambda1 <- 1
lambda2 <- 1
k_maxGrpNum <- 4
n_repls <- 5

#' This tibble contains all the parameter settings
parameter_settings_1D <- tibble(expand.grid(
  p = c(10, 50, seq(100, 1000, by = 100)) #, seq(2500, 10000, by = 2500))
))

# number of parameter settings
n_parameter_settings <- nrow(parameter_settings_1D)

# Create a progress bar
pb <- progress::progress_bar$new(total = n_parameter_settings)

#' Go over each of the parameter settings and check the run time
res <- lapply(1:n_parameter_settings, function(i) {
  
  # Get the parameters
  p <- as.integer(parameter_settings_1D[i, 'p'])
  
  # Create the random binary matrix weight matrix
  W <- band_matrix(p)
  
  # Create the corresponding connListObj needed by the flsa package
  connListObj <- create_connListObj(W)
  
  # generate the raw data 
  y <- rnorm(p)
  
  # Applies the flsa function to the data
  FLSA <- function(k = 4) {
    c(flsa::flsa(y, lambda1 = lambda1, lambda2 = lambda2, maxGrpNum = k * length(y)))
  }
  
  # Applies the wflsa package to the data
  wFLSA <- function() {
    wflsa::wflsa(y, W, lambda1 = lambda1, lambda2 = lambda2, offset = FALSE)$betas[[1]] # TODO: Why do we need `offest = FALSE`?
  }
  
  # Assesses the runtime 
  time <- microbenchmark::microbenchmark(wFLSA(), 
                                         FLSA(), times = n_repls)
  
  # Get the difference
  diff <- replicate(n_repls, mean((wFLSA() - FLSA())^2))
  pb$tick()
  
  return(list(time = time, difference = diff))
})

# Create plot
p_1d <- create_plot(lapply(res, function(x) x$time), 
                    parameter_settings = parameter_settings_1D, 
                    title = "1-D FLSA")

p_1_diff <- ggplot2::ggplot(data = data.frame(
  p = base::rep(parameter_settings_1D$p, each = n_repls),
  difference = unlist(lapply(res, function(x) x$difference))
), aes(x = p, y = difference)) + 
  geom_point() + 
  geom_smooth() + 
  ylab("Mean Squared Difference") + 
  xlab("p") + 
  ggtitle("Mean Squared Difference between wFLSA and FLSA") + 
  theme_minimal()

################################################################################
#                 SECTION 6: Heterogenous Image Smoothing
################################################################################

# Set seed for reproducibility
set.seed(42)

# Show the radial noise structure ----------------------------------------------
noise_structure <- get_noise_structure(400, 400)
dim(noise_structure) <- c(400, 400, 1)

image_ggplot(image_read(noise_structure)) +
  theme_void() +
  ggtitle("Radial noise structure") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))

# Show the original image and the image with added noise -----------------------

# Load image and resize
img_orig <- image_read("images/taylor_swift_free.png")
img_orig <- image_convert(image_resize(img_orig, "400x"), type = "Grayscale")
img <- as.integer(img_orig[[1]]) / 255

# Save image
image_write(image_read(img), "images/taylor_swift_original.png")

# Show image
image_ggplot(img_orig) +
  theme_void() +
  ggtitle("Original image") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))

# Add radial noise
noise_img <- add_noise(img, factor = 0.9)
image_ggplot(image_read(noise_img)) +
  theme_void() +
  ggtitle("Noisy image") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))

# Save image
image_write(image_read(noise_img), "images/taylor_swift_noisy.png")

# Smooth the image with wFLSA --------------------------------------------------

# Global attributes
w_height <- 5 # the patch height
w_width <- 4  # the patch width
lambda1 <- 0.001
lambda2 <- c(0.04, 0.1)
eps <- 1e-6

# Register parallel backend
registerDoParallel(cores = as.integer(parallel::detectCores() * 0.75))

# Patch-wise convolution attributes
height <- nrow(noise_img)
width <- ncol(noise_img)

# Get the noise structure and list of all patches indices
weights <- get_noise_structure(dim(noise_img)[1], dim(noise_img)[2])
idx <- expand.grid(i = seq(1, height - w_height + 1),
                   j = seq(1, width - w_width + 1))
idx <- lapply(seq_len(nrow(idx)), function(a) list(idx[a, 1], idx[a, 2]))

# Repeat for each lambda2
results <- list()
for (lambda in lambda2) {
  # Start time
  stime <- Sys.time()
  
  # Initialize the result matrices
  res_image <- array(0, dim = dim(noise_img))
  res_freq <- array(0, dim = dim(noise_img))
  
  # Apply wFLSA to each patch
  res <- foreach(idx = idx) %dopar% {
    # Get indices of the current patch
    idx_h <- idx[[1]]:min(height, idx[[1]] + w_height - 1)
    idx_w <- idx[[2]]:min(width, idx[[2]] + w_width - 1)
    
    # Get image and weight parts of the current patch
    sub_image <- noise_img[idx_h, idx_w, , drop = FALSE]
    sub_weights <- weights[idx_h, idx_w]
    
    # Apply function
    W <- generate_neighborhood_matrices(sub_weights, max(w_width, w_height))
    y <- c(sub_image)
    res <- wflsa::wflsa(y, W, lambda1, lambda, eps = eps, offset = FALSE)
    
    array(res$betas[[1]], dim = dim(sub_image))
  }
  
  # Combine the patches into the final image
  for (i in seq_along(res)) {
    # Get indices
    idx_h <- idx[[i]][[1]]:min(height, idx[[i]][[1]] + w_height - 1)
    idx_w <- idx[[i]][[2]]:min(width, idx[[i]][[2]] + w_width - 1)
    
    # Add image to results
    res_image[idx_h, idx_w, ] <- res_image[idx_h, idx_w, , drop = FALSE] + res[[i]]
    res_freq[idx_h, idx_w, ] <- res_freq[idx_h, idx_w, ] + 1
  }
  
  # Normalize the result
  res_image <- res_image / res_freq
  res_image <- (res_image - min(res_image)) / (max(res_image) - min(res_image))
  
  # Measure the time
  total_time <- Sys.time() - stime
  
  # Append results
  results[[as.character(lambda)]] <- list(
    image = res_image,
    time = total_time
  )
}


# Show results -----------------------------------------------------------------

# Show the result for lambda2 = 0.04
img <- results[[paste0(lambda2[1])]]$image
image_ggplot(image_read(img)) +
  theme_void() +
  ggtitle(paste0("Wflsa (lambda2 = ", lambda2[1], ")")) +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))
image_write(image_read(img), "images/taylor_swift_wflsa_1.png")
{ 
  cat("Runtime for lambda2 = 0.04:\n")
  print(results[[paste0(lambda2[1])]]$time)
}

# Show the result for lambda2 = 0.1
img <- results[[paste0(lambda2[2])]]$image
image_ggplot(image_read(img)) +
  theme_void() +
  ggtitle(paste0("Wflsa (lambda2 = ", lambda2[2], ")")) +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))
image_write(image_read(img), "images/taylor_swift_wflsa_2.png")
{ 
  cat("Runtime for lambda2 = 0.1:\n")
  print(results[[paste0(lambda2[2])]]$time)
}

# Comparison with Median Filter and NLM ----------------------------------------
library(reticulate)

# Create conda environment with the required packages
if (!("wflsa" %in% conda_list()$name)) {
  conda_create("wflsa", packages = c("conda-forge::opencv", "conda-forge::matplotlib"))
}
use_condaenv("wflsa")

# Apply Median Filter and NLM
py_run_file("cv2_denoiser.py")
