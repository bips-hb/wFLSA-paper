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
library(magick)
library(narray)
library(cli)
library(foreach)
library(parallel)
library(doParallel)
library(ggplot2)
library(grid)
library(gridExtra)
library(data.table)
library(batchtools)
library(geomtextpath)
library(ggthemes)

# Load utility functions
source("utils.R")

################################################################################
#                 Figure 2: Heterogenous Image Smoothing
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
img_orig <- image_read("figures/taylor_swift_free.png")
img_orig <- image_convert(image_resize(img_orig, "400x"), type = "Grayscale")
img <- as.integer(img_orig[[1]]) / 255

# Save image
image_write(image_read(img), "figures/taylor_swift_original.png")

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
image_write(image_read(noise_img), "figures/taylor_swift_noisy.png")

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
image_write(image_read(img), "figures/taylor_swift_wflsa_1.png")
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
image_write(image_read(img), "figures/taylor_swift_wflsa_2.png")
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
py_run_file("utils/cv2_denoiser.py")



################################################################################
#                       Figure 2: Runtime comparison
################################################################################

# Set seed
set.seed(42)

# Simulation parameters --------------------------------------------------------
n_repls <- 50
p <- c(10, 50, seq(100, 1000, by = 50))

# Algorithm parameters ---------------------------------------------------------
lambda1 <- c(0.1)
lambda2 <- c(0.1) 
eps <- 1e-10
k_maxGrpNum <- 15


# Registry ---------------------------------------------------------------------
reg_name <- "runtime"
reg_dir <- file.path("registries", reg_name)
dir.create("registries", showWarnings = FALSE)
unlink(reg_dir, recursive = TRUE)
makeExperimentRegistry(file.dir = reg_dir, 
                       packages = c("wflsa", "flsa", "microbenchmark"),
                       source = c("utils/utils.R", "utils/limit_cpus.R"),
                       conf.file = "utils/config.R")


# Problems ---------------------------------------------------------------------
problem_classic <- function(data, job, p) {
  W <- band_matrix(p)
  y <- rnorm(p) * p**0.75
  
  # Assume centered y
  y <- y - mean(y)
  
  list(y = y, W = W, connListObj = NULL, calc_flsa = TRUE)
}

problem_random <- function(data, job, p) {
  W <- random_weight_matrix(p)
  y <- rnorm(p) * p**0.75
  
  # Assume centered y
  y <- y - mean(y)
  
  list(y = y, W = W, connListObj = NULL, calc_flsa = FALSE)
}

problem_random_binary <- function(data, job, p) {
  W <- random_binary_weight_matrix(p, density = 0.5)
  connListObj <- create_connListObj(W) # needed by the flsa package
  y <- rnorm(p) * p**0.75
 
  # Assume centered y
  y <- y - mean(y)
  
  list(y = y, W = W, connListObj = connListObj, calc_flsa = TRUE)
}


# Add problems
addProblem(name = "random_full", fun = problem_random, seed = 1)
addProblem(name = "random_binary", fun = problem_random_binary, seed = 111)
addProblem(name = "classic", fun = problem_classic, seed = 1111)


# Algorithm --------------------------------------------------------------------

# Applies the wflsa package to the data
wFLSA <- function(y, W, l1, l2,  ...) {
  wflsa::wflsa(y, W, lambda1 = l1, lambda2 = l2, eps = 1e-10, ...)$betas[[1]]
}

algo <- function(data, job, instance, lambda1, lambda2) {
  y <- instance$y
  W <- instance$W
  
  # Calculate wFLSA
  time_wflsa <- microbenchmark::microbenchmark({
    res_wflsa = wFLSA(y, W, lambda1, lambda2)
  }, times = 1)$time / 1e9
  
  res <- data.frame(time = time_wflsa, method = "wflsa", p = length(y),
                    lambda1 = lambda1, lambda2 = lambda2)
  
  res
}

# Add algorithm
addAlgorithm(name = "solver", fun = algo)

# Experiments ------------------------------------------------------------------
prob_design <- list(
  random_full = expand.grid(p = p),
  random_binary = expand.grid(p = p),
  classic = expand.grid(p = p)
)

algo_design <- list(
  solver = expand.grid(
    lambda1 = lambda1, 
    lambda2 = lambda2)
)

addExperiments(prob_design, algo_design, repls = n_repls)
summarizeExperiments()
testJob(1)

# Submit -----------------------------------------------------------------------
ids <- findNotSubmitted()$job.id
submitJobs(ids = sample(ids))
waitForJobs()

# Results ----------------------------------------------------------------------
loadRegistry(file.dir = reg_dir, conf.file = "utils/config.R")

res <- flatten(reduceResultsDataTable())
args <- getJobPars()[, c("job.id", "problem")]
dt_time <- merge(res, args, by = "job.id")

# Set levels and labels
dt_time$problem <- factor(dt_time$problem, 
                          levels = c("classic", "random_binary", "random_full"),
                          labels = c("Classic Problem", "Random binary W", "Random sparse W"))
dt_time$lambda1 <- paste0("lambda[1]==", dt_time$lambda1)
dt_time$lambda2 <- paste0("lambda[2]==", dt_time$lambda2)

# Mean time over replications
dt_time <- dt_time[, .(time = mean(time)), by = .(p, lambda1, lambda2, problem)]

# Plot results -----------------------------------------------------------------
ggplot(dt_time, aes(x = p, y = time, color = problem)) +
  geom_texthline(yintercept = 1, color = "gray25", linetype = "dashed", 
                 label = "1 second threshold", hjust = 0.75) +
  geom_point() +
  geom_line() +
  facet_grid(lambda1 ~ lambda2, labeller = label_parsed) +
  scale_color_colorblind() +
  theme_minimal() +
  scale_y_log10() +
  theme(legend.position = "top") +
  labs(x = "p", y = "Time (s)", shape = "Package", color = "Problem")

# Check if directory exists
if (!dir.exists("figures")) dir.create("figures")

ggsave("figures/runtime_comparison.pdf", width = 9, height = 5)
ggsave("figures/runtime_comparison.png", width = 9, height = 5, dpi = 400)



