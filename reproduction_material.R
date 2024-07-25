################################################################################
#                     Reproduction material for the paper:
#
#     An Alternating Direction Method of Multipliers Algorithm for the 
#                Weighted Fused LASSO  Signal Approximator
#
#               SECTION 6: Heterogenous Image Smoothing
################################################################################

# Load required libraries
library(magick)
library(narray)
library(wflsa)
library(foreach)
library(doParallel)
library(ggplot2)
library(grid)
library(gridExtra)

# Set seed for reproducibility
set.seed(42)


# Utility functions ------------------------------------------------------------

# Method for calculating the noise intensity which increases radially
get_noise_structure <- function(height, width) {
  # Calculate the maximal distance from the midpoint to the corners
  max_dist <- sqrt((width / 2 - 0.5)^2 + (height / 2 - 0.5)^2)
  
  # Create a grid of coordinates
  y_coords <- matrix(base::rep(1:height, each = width), ncol = width, byrow = TRUE)
  x_coords <- matrix(base::rep(1:width, height), ncol = width, byrow = TRUE)
  
  # Calculate the distance from the midpoint
  dist_from_center <- sqrt((y_coords - 0.5 - height / 2)^2 + (x_coords - 0.5 - width / 2)^2)
  
  # Calculate the noise intensity between 0 and 1 with a radial increase
  noise_structure <- (dist_from_center / max_dist)^2
  
  noise_structure
}

# Method for adding noise to an image
add_noise <- function(image, factor = 0.5) {
  height <- nrow(image)
  width <- ncol(image)
  
  # Get the noise intensity
  noise_strength <- get_noise_structure(height, width) * factor
  
  # Create noise (random values between -1 and 1)
  noise <- array(runif(height * width, min = -1, max = 1), dim = c(height, width))
  additiv_noise <- noise * noise_strength
  dim(additiv_noise) <- c(dim(additiv_noise), 1)
  
  # Add noise to the image and clip the values to [0, 1]
  res <- image + additiv_noise
  res[res < 0] <- 0
  res[res > 1] <- 1
  
  res
}

# This method converts the matrix with the noise intensities for each pixel 
# into the weight matrix used for wFLSA, considering only the neighboring 
# `num_neighbors`. Thus, for an image of size HxW, the weight matrix will have
# the shape (H*W)x(H*W).
generate_neighborhood_matrices <- function(weight_matrix, num_neighbors = 5) {
  height <- nrow(weight_matrix)
  width <- ncol(weight_matrix)
  
  neighbor_matrices <- array(0, dim = c(height, width, 1, height,  width, 1))
  
  for (i in 1:height) {
    for (j in 1:width) {
      neighbors <- array(0, dim = c(height, width, 1))
      row_idx <- max(0, i - num_neighbors):min(height, i + num_neighbors)
      col_idx <- max(0, j - num_neighbors):min(width, j + num_neighbors)
      weights <- weight_matrix[row_idx, col_idx]
      dim(weights) <- c(dim(weights), 1)
      neighbors[row_idx, col_idx, ] <- narray::rep(weights, 1, 3)
      neighbors[i, j, 1] <- 0
      neighbor_matrices[i, j, 1, , , ] <- neighbors
    }
  }
  
  dim(neighbor_matrices) <- c(height * width, height * width)
  neighbor_matrices
}

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
w_height <- 4 # the patch height
w_width <- 5  # the patch width
lambda1 <- 0.001
lambda2 <- c(0.04, 0.1)
eps <- 1e-6

# Register parallel backend
registerDoParallel(cores = parallel::detectCores() %/% 2)

# Patch-wise convolution attributes
height <- nrow(noise_img)
width <- ncol(noise_img)

# Get the noise structure and list of all patches indices
weights <- get_noise_structure(dim(noise_img)[1], dim(noise_img)[2])
idx <- expand.grid(i = seq(1, height - w_height + 1),
                   j = seq(1, width - w_width + 1))
idx <- lapply(seq_len(nrow(idx)), function(a) list(idx[a, 1], idx[a, 2]))

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
  res <- wflsa::wflsa(y, W, lambda1, lambda2, eps = eps, offset = FALSE)

  list(
    first_lambda = array(res$betas[[1]], dim = dim(sub_image)),
    second_lambda = array(res$betas[[2]], dim = dim(sub_image))
  )
}

# Combine the patches into the final image
res_first_lambda <- array(0, dim = dim(noise_img))
res_second_lambda <- array(0, dim = dim(noise_img))
res_freq <- array(0, dim = dim(noise_img))
for (i in seq_along(res)) {
  # Get indices
  idx_h <- idx[[i]][[1]]:min(height, idx[[i]][[1]] + w_height - 1)
  idx_w <- idx[[i]][[2]]:min(width, idx[[i]][[2]] + w_width - 1)

  # Add image to results
  res_first_lambda[idx_h, idx_w, ] <-
    res_first_lambda[idx_h, idx_w, , drop = FALSE] + res[[i]]$first_lambda
  res_second_lambda[idx_h, idx_w, ] <-
    res_second_lambda[idx_h, idx_w, , drop = FALSE] + res[[i]]$second_lambda
  res_freq[idx_h, idx_w, ] <- res_freq[idx_h, idx_w, ] + 1
}
res_first_lambda <- res_first_lambda / res_freq
res_first_lambda <- (res_first_lambda - min(res_first_lambda)) /
  (max(res_first_lambda) - min(res_first_lambda))
res_second_lambda <- res_second_lambda / res_freq
res_second_lambda <- (res_second_lambda - min(res_second_lambda)) /
  (max(res_second_lambda) - min(res_second_lambda))

# Show results -----------------------------------------------------------------

# Show the result for lambda2 = 0.04
image_ggplot(image_read(res_first_lambda)) +
  theme_void() +
  ggtitle("Wflsa (lambda2 = 0.04)") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))
image_write(image_read(res_first_lambda), "images/taylor_swift_wflsa_1.png")

# Show the result for lambda2 = 0.1
image_ggplot(image_read(res_second_lambda)) +
  theme_void() +
  ggtitle("Wflsa (lambda2 = 0.1)") +
  theme(plot.title = element_text(hjust = 0.5, size = 15, face = "bold"))
image_write(image_read(res_second_lambda), "images/taylor_swift_wflsa_2.png")


# Comparison with Median Filter and NLM ----------------------------------------
library(reticulate)

# Create conda environment with the required packages
if (!("wflsa" %in% conda_list()$name)) {
  conda_create("wflsa", packages = c("conda-forge::opencv", "matplotlib"))
}
use_condaenv("wflsa")

# Apply Median Filter and NLM
py_run_file("cv2_denoiser.py")
