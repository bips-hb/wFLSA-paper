library(data.table)
library(batchtools)
library(ggplot2)
library(flsa)
library(wflsa)
library(microbenchmark)
library(geomtextpath)
library(ggthemes)

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
problem_random <- function(data, job, p) {
  W <- random_weight_matrix(p)
  y <- rnorm(p) * p**0.75
  
  # Both assume centered y
  y <- y - mean(y)
  
  list(y = y, W = W, connListObj = NULL, calc_flsa = FALSE)
}

problem_random_binary <- function(data, job, p) {
  W <- random_binary_weight_matrix(p, density = 0.5)
  connListObj <- create_connListObj(W) # needed by the flsa package
  y <- rnorm(p) * p**0.75
  
  # Both assume centered y
  y <- y - mean(y)
  
  list(y = y, W = W, connListObj = connListObj, calc_flsa = TRUE)
}

problem_classic <- function(data, job, p) {
  W <- band_matrix(p)
  y <- rnorm(p) * p**0.75
  
  # Both assume centered y
  y <- y - mean(y)
  
  list(y = y, W = W, connListObj = NULL, calc_flsa = TRUE)
}

# Add problems
addProblem(name = "random_full", fun = problem_random, seed = 1)
addProblem(name = "random_binary", fun = problem_random_binary, seed = 111)
addProblem(name = "classic", fun = problem_classic, seed = 1111)


# Algorithm --------------------------------------------------------------------
# Methods to be compared
FLSA <- function(y, l1, l2, maxGrpNum, eps, connListObj = list(connList_is_null = TRUE)) {
  # if there are no connections what so ever
  if (connListObj$connList_is_null) {
    c(flsa::flsa(y, lambda1 = l1, lambda2 = l2, thr = eps)) 
  } else {
    flsa_obj <- flsa::flsa(y, lambda1 = l1, connListObj = connListObj$connList, 
                           maxGrpNum = maxGrpNum, thr = eps)
    c(flsaGetSolution(flsa_obj, lambda1 = l1, lambda2 = l2))
  }
}

# Applies the wflsa package to the data
wFLSA <- function(y, W, l1, l2, eps,  ...) {
  wflsa::wflsa(y, W, lambda1 = l1, lambda2 = l2, eps = eps, ...)$betas[[1]]
}

algo <- function(data, job, instance, lambda1, lambda2, eps, k_maxGrpNum) {
  y <- instance$y
  W <- instance$W
  
  # Calculate wFLSA
  time_wflsa <- microbenchmark::microbenchmark({
    res_wflsa = wFLSA(y, W, lambda1, lambda2, eps = eps)
  }, times = 1)$time / 1e9
  
  df_time <- data.frame(time = time_wflsa, method = "wflsa", p = length(y),
                        lambda1 = lambda1, lambda2 = lambda2, eps = eps)
  df_res <- data.frame(beta = res_wflsa, method = "wflsa", p = length(y),
                        lambda1 = lambda1, lambda2 = lambda2, eps = eps)
   
  
  if (instance$calc_flsa) {
    connListObj <- instance$connListObj
    
    if (is.null(connListObj)) {
      connListObj <- list(connList_is_null = TRUE)
    }
    
    # Calculate FLSA
    time_flsa <- microbenchmark::microbenchmark({
      res_flsa = FLSA(y, lambda1, lambda2, maxGrpNum = k_maxGrpNum * length(y), 
                      eps = eps, connListObj = connListObj)
    }, times = 1)$time / 1e9
    
    df_time <- rbind(df_time,
                     data.frame(time = time_flsa, method = "flsa", p = length(y),
                                lambda1 = lambda1, lambda2 = lambda2, eps = eps))
    df_res <- rbind(df_res, 
                    data.frame(beta = res_flsa, method = "flsa", p = length(y),
                               lambda1 = lambda1, lambda2 = lambda2, eps = eps))
  }
  
  list(time = df_time, res = df_res)
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
    lambda2 = lambda2, 
    eps = eps, 
    k_maxGrpNum = k_maxGrpNum)
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

res <- reduceResultsDataTable()
args <- getJobPars()[, c("job.id", "problem")]
dt_time <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  cbind(job.id = res$job.id[i], res$result[[i]]$time)
  }))
dt_res <- rbindlist(lapply(seq_len(nrow(res)), function(i) {
  if (!is.null(res$result[[i]]$res)) {
    cbind(job.id = res$job.id[i], res$result[[i]]$res)
  } else {
    NULL
  }
}))

dt_time <- merge(dt_time, args, by = "job.id")
dt_res <- merge(dt_res, args, by = "job.id")

# Set levels and labels
dt_res$method <- factor(dt_res$method, levels = c("wflsa", "flsa"),
                        labels = c("wFLSA (ours)", "FLSA"))
dt_res$problem <- factor(dt_res$problem, 
                         levels = c("classic", "random_binary", "random_full"),
                         labels = c("Classic Problem", "Random binary W", "Random W"))
dt_res$lambda1 <- paste0("lambda[1]==", dt_res$lambda1)
dt_res$lambda2 <- paste0("lambda[2]==", dt_res$lambda2)

dt_time$method <- factor(dt_time$method, levels = c("wflsa", "flsa"),
                        labels = c("wFLSA (ours)", "FLSA"))
dt_time$problem <- factor(dt_time$problem, 
                         levels = c("classic", "random_binary", "random_full"),
                         labels = c("Classic Problem", "Random binary W", "Random W"))
dt_time$lambda1 <- paste0("lambda[1]==", dt_time$lambda1)
dt_time$lambda2 <- paste0("lambda[2]==", dt_time$lambda2)


# Save results -----------------------------------------------------------------
if (!file.exists("results")) dir.create("results")
saveRDS(dt_time, file = "results/runtime_comparison.rds")
saveRDS(dt_res, file = "results/betas.rds")

# To reproduce the final results:
# dt_time <- readRDS("results/runtime_comparison.rds")
# dt_res <- readRDS("results/betas.rds")

# Plot results -----------------------------------------------------------------
if (!file.exists("figures")) dir.create("figures")

# Time Comparison
dt_time <- dt_time[, .(time = mean(time)), 
                   by = .(method, p, lambda1, lambda2, eps, problem)]
ggplot(dt_time, aes(x = p, y = time, color = problem, linetype = method,shape = method)) +
  geom_texthline(yintercept = 1, color = "gray25", linetype = "dashed", 
                 label = "1 second threshold", hjust = 0.75) +
  geom_texthline(yintercept = 60, color = "gray25", linetype = "dashed", 
                 label = "1 minute threshold", hjust = 0.6) +
  geom_point() +
  geom_line() +
  #facet_grid(lambda1 ~ lambda2, labeller = label_parsed) +
  scale_color_colorblind() +
  theme_minimal() +
  scale_y_log10() +
  theme(legend.position = "top") +
  labs(x = "p", y = "Time (s)", shape = "Package", linetype = "Package",
       color = "Problem")

ggsave("figures/runtime_comparison_full.pdf", width = 9, height = 5)
ggsave("figures/runtime_comparison_full.png", width = 9, height = 5, dpi = 400)


dt_time <- dt_time[method == "wFLSA (ours)", ]
ggplot(dt_time, aes(x = p, y = time, color = problem)) +
  geom_texthline(yintercept = 1, color = "gray25", linetype = "dashed", 
                 label = "1 second threshold", hjust = 0.75) +
  geom_texthline(yintercept = 60, color = "gray25", linetype = "dashed", 
                 label = "1 minute threshold", hjust = 0.6) +
  geom_point() +
  geom_line() +
  #facet_grid(lambda1 ~ lambda2, labeller = label_parsed) +
  scale_color_colorblind() +
  theme_minimal() +
  scale_y_log10() +
  theme(legend.position = "top") +
  labs(x = "p", y = "Time (s)", shape = "Package", color = "Problem")

ggsave("figures/runtime_comparison.pdf", width = 9, height = 5)
ggsave("figures/runtime_comparison.png", width = 9, height = 5, dpi = 400)

# Norm of results
dt_norm <- dt_res[, .(norm = sqrt(mean(beta^2))), 
                  by = .(method, p, lambda1, lambda2, eps, problem)]
ggplot(dt_norm, aes(x = p, y = norm, color = problem)) +
  geom_point() +
  geom_line() +
  scale_color_colorblind() +
  #facet_grid(lambda1 ~ lambda2, labeller = label_parsed) +
  theme_minimal() +
  scale_y_log10() +
  ggtitle("Eucledean Norm (normalized by p)") +
  theme(legend.position = "top") +
  labs(x = "p", y = "Euclidean Norm (normalized by p)", color = "Algorithm")
ggsave("figures/euclidean_norm.pdf", width = 9, height = 5)


# Difference between results
dt_diff <- dt_res[method == "wFLSA (ours)" & problem != "Random W" , -"method"]
dt_diff$beta <- dt_diff$beta - dt_res[method == "FLSA" & problem != "Random W", ]$beta
colnames(dt_diff)[2] <- "beta_diff"
dt_error <-  dt_diff[, .(error = sqrt(mean(beta_diff**2))), 
                     by = .(p, lambda1, lambda2, eps, problem, job.id)]
ggplot(dt_error, aes(x = as.factor(p), y = error)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = 1e-9, fill = "gray", alpha = 0.5) +
  geom_texthline(yintercept = 1e-9, label  = "Threshold (1e-9)",
                 linetype = "dashed", color = "black") +
  geom_boxplot(aes(fill = problem), outlier.size = 0.4) +
  scale_fill_manual(values = c("#E69F00", "#56B4E9")) +
  #facet_grid(lambda1 ~ lambda2, labeller = label_parsed) +
  scale_y_log10(limits = c(-Inf, 1)) +
  theme_minimal() +
  theme(legend.position = "top") +
  labs(x = "Number of p", y = "Mean squared difference of betas",
       fill = "Problem")
ggsave("figures/MSE_flsa_vs_wflsa.pdf", width = 9, height = 5)
ggsave("figures/MSE_flsa_vs_wflsa.png", width = 9, height = 5, dpi = 400)




