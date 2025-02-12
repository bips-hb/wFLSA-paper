################################################################################
#                           UTILITY FUNCTIONS
################################################################################

# Runtime comparison -----------------------------------------------------------
#' Util. Function for Runtime Analysis of the FLSA and wFLSA Package
#'
#' This script contains various utility functions used to check the runtime 
#' of the FLSA and WFLSA packages. It includes functions for creating a 
#' connListObj for the FLSA package, copying the upper triangle of a square 
#' matrix to its lower triangle, and generating a band matrix with a band around 
#' the diagonal.
#' 

#' Create a connListObj for FLSA package
#' 
#' This function creates a connListObj for the FLSA package given a weight matrix.
#' 
#' @param W A weight matrix.
#' @return A connListObj for FLSA package and a logical value indicating whether 
#'         all elements of connList are NULL.
create_connListObj <- function(W) {
  m <- nrow(W)
  
  connList <- vector("list", m)
  class(connList) <- "connListObj"
  
  lapply(1:m, function(i) {
    indices <- which(W[i, ] != 0)
    if (length(indices) != 0) {
      connList[[i]] <<- as.integer(which(W[i, ] != 0) - 1)
    }
  })
  
  names(connList) <- as.character(0:(m-1))
  
  connList_is_null <- all(sapply(connList, function(l) is.null(l)))
  
  return(list(connList = connList, connList_is_null = connList_is_null))
}


#' Copy Upper Triangle to Lower Triangle
#'
#' Copies the upper triangle of a square matrix to its lower triangle.
#'
#' @param mat A square matrix.
#' @return A matrix with the upper triangle copied to the lower triangle.
#' @export
#'
#' @examples
#' mat <- matrix(1:9, nrow = 3)
#' copy_upper_to_lower(mat)
copy_upper_to_lower <- function(mat) {
  p <- nrow(mat)
  for (i in 1:(p - 1)) {
    mat[(i + 1):p, i] <- mat[i, (i + 1):p]
  }
  return(mat)
}

#' Band Matrix Generator
#'
#' \code{band_matrix} function creates a square matrix with a band around the diagonal.
#'
#' @param p Size of the square matrix.
#' @return A square matrix with a band around the diagonal.
#' @usage band_matrix(p)
#' Default band width is set to 1, but it can be adjusted as needed.
#'
#' @examples
#' \dontrun{
#' # Generate a band matrix of size 5
#' band_matrix(5)
#' }
band_matrix <- function(p) {
  
  # Create a matrix with a band around the diagonal
  my_matrix <- matrix(0, nrow = p, ncol = p)
  
  for (i in 1:p) {
    lower <- max(1, i - 1)
    upper <- min(p, i + 1)
    my_matrix[i, lower:upper] <- 1
  }
  
  diag(my_matrix) <- 0  # Set diagonal elements to 0 
  my_matrix <- copy_upper_to_lower(my_matrix)
  return(my_matrix)
}

#' Generate Random Binary Weight Matrix
#'
#' \code{random_binary_weight_matrix} function creates a random binary weight matrix 
#' of size \code{p} with a specified density.
#'
#' @param p Size of the square weight matrix.
#' @param density Density of the binary values (probability of being 1).
#' @return A random binary weight matrix of size \code{p} with the specified density.
#'
#' @examples
#' \dontrun{
#' # Generate a random binary weight matrix of size 5 with density 0.3
#' random_binary_weight_matrix(5, 0.3)
#' }
random_binary_weight_matrix <- function(p, density) {
  W <- matrix(rbinom(p*p, 1, density), nrow = p)
  W[upper.tri(W)] <- t(W)[upper.tri(W)]
  diag(W) <- 0
  return(W)
}

#' Generate Random Weight Matrix
#'
#' \code{random_weight_matrix} function creates a random weight matrix 
#' of size \code{p} based on a gaussian distribution.
#'
#' @param p Size of the square weight matrix.
#' @return A random weight matrix of size \code{p}.
#'
#' @examples
#' \dontrun{
#' # Generate a random weight matrix of size 5
#' random_binary_weight_matrix(5, 0.3)
#' }
random_weight_matrix <- function(p) {
  W <- matrix(runif(p*p) * rbinom(p*p, 1, 0.5), nrow = p)
  W[upper.tri(W)] <- t(W)[upper.tri(W)]
  diag(W) <- 0
  return(W)
}

#' Create Plot for Runtime Assessment Results
#'
#' \code{create_plot} function reads in the runtime assessment results from a file, 
#' processes the data, and creates a plot to visualize the performance of different 
#' algorithms based on the parameter settings.
#'
#' @param res The runtime assessment results.
#' @param parameter_settings A tibble containing the parameter settings used for the assessment.
#' @param title A title for the plot.
#' @return A plot visualizing the performance of different algorithms based on the parameter settings.
create_plot <- function(res, 
                        parameter_settings = parameter_settings, 
                        title = "") {
  
  # go over all the results for the different parameter settings
  res <- lapply(1:length(res), function(i) {
    
    df <- as.data.frame(res[[i]])
    df$expr <- as.character(df$expr)
    df$expr <- substring(df$expr, 1, nchar(df$expr) - 2)
    colnames(df) <- c("algorithm", "time")
    
    df %>% mutate(p = as.integer(parameter_settings[i, 1]))
    #density = as.double(parameter_settings[i,2]))
  })
  
  # combine all the results into a single tibble
  res <- do.call(rbind, res)
  
  # turn the time data into seconds
  res <- res %>% mutate(time = time / 1e9)
  
  # calculate mean time for each parameter setting
  res <- res %>% group_by(p, algorithm) %>% 
    summarise(time = mean(time))
  
  # create the plot
  ggplot2::ggplot(res, aes(x = p, y = time, color = algorithm)) + 
    geom_line() +
    geom_point() +
    ylab("time (s)") + 
    ggtitle(title) + 
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_log10(expand = c(0, 0)) +
    theme_bw() + 
    guides(color=guide_legend(override.aes=list(fill=NA)))
}


# Get results as data.frame
get_results <- function(res, parameter_settings, var = "time", label = NULL) {
  df <- lapply(seq_along(res), function(i) {
    df <- as.data.frame(res[[i]][[var]])
    
    if (var == "time") {
      df$expr <- as.character(df$expr)
      df$expr <- substring(df$expr, 1, nchar(df$expr) - 2)
      colnames(df) <- c("algorithm", "time")
      df$time <- df$time / 1e9
    } else if (var == "difference") {
      colnames(df) <- c("MSE")
    }
    df$problem <- if (is.null(label)) NULL else gsub("^\\s+|[()]", "", label)
    df$label <- if (is.null(label)) NULL else paste0(df$algorithm, label)
    df$p <- as.integer(parameter_settings[i, 1])
    df$lambda1 <- as.double(parameter_settings[i, 2])
    df$lambda2 <- as.double(parameter_settings[i, 3])
    
    df
  })
  
  do.call(rbind, df)
}


# Image Smoothing --------------------------------------------------------------

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