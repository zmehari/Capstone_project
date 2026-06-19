##########################################################
# Movie recommendation system script
##########################################################

# Create edx and final_holdout_test sets
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)

colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################
# RMSE function
##########################################################
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##########################################################
# Baseline model: mean only
##########################################################
mu <- mean(edx$rating)

pred_mean <- rep(mu, nrow(final_holdout_test))
rmse_mean <- RMSE(final_holdout_test$rating, pred_mean)

##########################################################
# Movie effect model
##########################################################
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu), .groups = "drop")

pred_movie <- final_holdout_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i)

pred_movie$pred[is.na(pred_movie$pred)] <- mu

rmse_movie <- RMSE(final_holdout_test$rating, pred_movie$pred)

##########################################################
# Movie + user effect model (unregularized)
##########################################################
user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i), .groups = "drop")

pred_movie_user <- final_holdout_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)

pred_movie_user$pred[is.na(pred_movie_user$pred)] <- mu

rmse_movie_user <- RMSE(final_holdout_test$rating, pred_movie_user$pred)

##########################################################
# Regularized model (tuning using internal edx split)
##########################################################
set.seed(1)

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_temp <- edx[-test_index, ]
validation_temp <- edx[test_index, ]

validation_temp <- validation_temp %>%
  semi_join(train_temp, by = "movieId") %>%
  semi_join(train_temp, by = "userId")

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){

  mu <- mean(train_temp$rating)

  b_i <- train_temp %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l), .groups = "drop")

  b_u <- train_temp %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + l), .groups = "drop")

  preds <- validation_temp %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)

  preds[is.na(preds)] <- mu

  RMSE(validation_temp$rating, preds)
})

best_lambda <- lambdas[which.min(rmses)]
rmse_validation_regularized <- min(rmses)

##########################################################
# Train final regularized model on full edx
##########################################################
mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + best_lambda), .groups = "drop")

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + best_lambda), .groups = "drop")

pred_regularized <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)

pred_regularized$pred[is.na(pred_regularized$pred)] <- mu

rmse_regularized <- RMSE(final_holdout_test$rating, pred_regularized$pred)

##########################################################
# Final comparison table
##########################################################
results <- tibble(
  Model = c("Mean only",
            "Movie effect",
            "Movie + User effect",
            "Regularized (tuned on edx)"),

  RMSE_edx_validation = c(NA, NA, NA, rmse_validation_regularized),

  RMSE_final_holdout_test = c(rmse_mean,
                              rmse_movie,
                              rmse_movie_user,
                              rmse_regularized)
)

results %>% arrange(RMSE_final_holdout_test)
