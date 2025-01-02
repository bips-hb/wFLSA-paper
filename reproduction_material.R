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
library(cli)
library(foreach)
library(parallel)
library(pbmcapply)
library(doParallel)
library(ggplot2)
library(grid)
library(gridExtra)

# Load utility functions
source("utils.R")

################################################################################
#                       Section ???: Runtime comparison
################################################################################

# Random graph -----------------------------------------------------------------
# Note: The package flsa does not support the weighted fused lasso. Therefore,
# we can only compare the runtime of the wflsa package.

# Set seed for reproducibility
set.seed(42)

# Constants
lambda1 <- 0.1
lambda2 <- 0.1
n_repls <- 2#0
num_threads <- 1
eps <- 1e-9

#' This tibble contains all the parameter settings. It includes combinations of 
#' 'p' (size of the square weight matrix) and 'density' (density of the binary values).
parameter_settings <- tibble(expand.grid(
  p = c(1000, 5000) #c(10, 50, seq(100, 1000, by = 100), 1250, 1500, 1750, 2000, 3000, 4000, 5000)
))

# number of parameter settings
n_parameter_settings <- nrow(parameter_settings)

#' Go over each of the parameter settings and check the run time
res <- lapply(cli_progress_along(1:n_parameter_settings, "Runtime: Random Graph"), function(i) {
  
  # Get the parameters
  p <- as.integer(parameter_settings[i, 'p'])
  
  # Create the random binary matrix weight matrix
  W <- random_weight_matrix(p)
  
  # generate the raw data 
  y <- rnorm(p)
  
  # Applies the wflsa package to the data
  wFLSA <- function() {
    wflsa::wflsa(y, W, lambda1 = lambda1, lambda2 = lambda2, eps = eps)$betas[[1]]
  }
  
  # Assesses the runtime 
  time <- microbenchmark::microbenchmark(wFLSA(), times = n_repls)
  
  return(list(time = time))
}) #, mc.cores = num_threads)

# Get results as data.frame
res_random_graph <- get_results(res, parameter_settings, label = " (Random Graph)")

# Save results
if (!dir.exists("results")) dir.create("results")
saveRDS(res_random_graph, "results/random_graph.rds")

# Random binary graph ----------------------------------------------------------
# Set seed for reproducibility
set.seed(42)

# Constants
lambda1 <- 0.1
lambda2 <- 0.1
k_maxGrpNum <- 12 # Needs to be high, flsa fails to converge otherwise
eps <- 1e-9
n_repls <- 1#0
num_threads <- 1

#' This tibble contains all the parameter settings. It includes combinations of 
#' 'p' (size of the square weight matrix) and 'density' (density of the binary values).
parameter_settings <- tibble(expand.grid(
  p = c(2500), #, c(10, 50, seq(100, 1000, by = 100), 1250, 1500, 1750, 2000, 3000, 4000, 5000), 
  density = c(.5)
))

# number of parameter settings
n_parameter_settings <- nrow(parameter_settings)

#' Go over each of the parameter settings and check the run time
res <- lapply(cli_progress_along(1:n_parameter_settings, "Runtime: Random Binary Graph"), function(i) {
  
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
      c(flsa::flsa(y, lambda1 = lambda1, lambda2 = 0, thr = eps))
    } else {
      flsa_obj <- flsa::flsa(y, lambda1 = lambda1, connListObj = connListObj$connList, 
                             thr = eps, maxGrpNum = k_maxGrpNum * length(y))
      c(flsaGetSolution(flsa_obj, lambda1 = lambda1, lambda2 = lambda2))
    }
  }
  
  # Applies the wflsa package to the data
  wFLSA <- function() {
    wflsa::wflsa(y, W, lambda1 = lambda1, lambda2 = lambda2, eps = eps, offset = FALSE)$betas[[1]]
  }
  
  # Assesses the runtime 
  time <- microbenchmark::microbenchmark(wFLSA(), FLSA(), times = n_repls)
  
  # Get the difference
  diff <- replicate(n_repls, mean((wFLSA() - FLSA())^2))
  
  return(list(time = time, difference = diff))
}) #, mc.cores = num_threads)

res_random_binary_graph <- get_results(res, parameter_settings, label = " (Random Binary Graph)")
res_random_binary_graph_mse <- get_results(res, parameter_settings, 
                                           var = "difference", label = "Random Binary Graph")

# Save results
if (!dir.exists("results")) dir.create("results")
saveRDS(res_random_binary_graph, "results/random_binary_graph.rds")
saveRDS(res_random_binary_graph_mse, "results/random_binary_graph_mse.rds")

# Classic 1-D FLSA -------------------------------------------------------------
# Set seed for reproducibility
set.seed(42)

# Constants
lambda1 <- 0.1
lambda2 <- 0.1
k_maxGrpNum <- 4
eps <- 1e-9
n_repls <- 2#0
num_threads <- 1

#' This tibble contains all the parameter settings
parameter_settings_1D <- tibble(expand.grid(
  p = c(1000, 5000) #c(10, 50, seq(100, 1000, by = 100), 1250, 1500, 1750, 2000, 3000, 4000, 5000)
))

# number of parameter settings
n_parameter_settings <- nrow(parameter_settings_1D)

#' Go over each of the parameter settings and check the run time
res <- lapply(cli_progress_along(1:n_parameter_settings, "Runtime: Classic 1D"), function(i) {
  
  # Get the parameters
  p <- as.integer(parameter_settings_1D[i, 'p'])
  
  # Create the random binary matrix weight matrix
  W <- band_matrix(p)
  
  # generate the raw data 
  y <- rnorm(p)
  
  # Applies the flsa function to the data
  FLSA <- function() {
    c(flsa::flsa(y, lambda1 = lambda1, lambda2 = lambda2, 
                 maxGrpNum = k_maxGrpNum * length(y), thr = eps))
  }
  
  # Applies the wflsa package to the data
  wFLSA <- function() {
    wflsa::wflsa(y, W, lambda1 = lambda1, lambda2 = lambda2, 
                 eps = eps, offset = FALSE)$betas[[1]] # TODO: Why do we need `offest = FALSE`?
  }
  
  # Assesses the runtime 
  time <- microbenchmark::microbenchmark(wFLSA(), FLSA(), times = n_repls)
  
  # Get the difference
  diff <- replicate(n_repls, mean((wFLSA() - FLSA())^2))
  
  return(list(time = time, difference = diff))
}) #, mc.cores = num_threads)

# Get results as data.frame
res_1D <- get_results(res, parameter_settings_1D, label = " (classic 1D)")
res_1D_mse <- get_results(res, parameter_settings_1D, var = "difference", label = "Classic 1D")

# Save results
if (!dir.exists("results")) dir.create("results")
saveRDS(res_1D, "results/classic_1D.rds")
saveRDS(res_1D_mse, "results/classic_1D_mse.rds")


# Create plots -----------------------------------------------------------------

# Load results for time comparison
res_time <- rbind(
  readRDS("results/random_graph.rds"),
  readRDS("results/random_binary_graph.rds"),
  readRDS("results/classic_1D.rds")
)

res_time <- res_time %>% 
  group_by(algorithm, label, p, problem) %>% 
  summarise(mean_time = mean(time)) %>%
  mutate(label = factor(label, levels = 
                          c("FLSA (classic 1D)", "FLSA (Random Binary Graph)", 
                            "wFLSA (classic 1D)", "wFLSA (Random Binary Graph)", 
                            "wFLSA (Random Graph)")))

# Create plot for time comparison
p_time <- ggplot(res_time, aes(x = p, y = mean_time, color = label)) + 
  geom_line(linewidth = 0.25) +
  geom_point() +
  labs(y = "time (s)", color = "Method") + 
  ggtitle("Runtime Comparison") + 
  facet_grid(cols = vars(problem)) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_log10(expand = c(0, 0)) +
  theme_bw() + 
  guides(color=guide_legend(override.aes=list(fill=NA)))


# Load results for MSE comparison
res_mse <- rbind(
  readRDS("results/random_binary_graph_mse.rds"),
  readRDS("results/classic_1D_mse.rds")
)

ggplot(res_mse, aes(x = p, y = MSE, color = label)) + 
  geom_rect(xmin = -Inf, xmax = Inf, ymin = 0, ymax = 1e-10, color = "darkgray", fill = "grey", alpha = 0.5, data = NULL) +
  geom_point() +
  labs(y = "MSE", color = "Problem") + 
  ggtitle("Mean Squared Error Comparison") + 
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_log10() +
  ylim(0, 5e-10) +
  theme_bw() + 
  guides(color=guide_legend(override.aes=list(fill=NA)))

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
