library(tidyverse)
library(tidytext)
library(tidymodels)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(sparsesvd)
library(glmnet)

source("../scripts/modified-preprocessing.R")
source("./scripts/projection.R")

# load raw data
load('../data/claims-raw.RData')

# clean raw data
claims_clean <- claims_raw %>%
  parse_data()

# apply nlp to get DTM with labels
claims <- claims_clean %>% 
  nlp_fn()

set.seed(11182024)

# partition claims data
partitions <- claims %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
test_labels <- testing(partitions) %>%
  select(.id, bclass, mclass)
test_id <- test_labels %>% pull(.id)
save(test_id, file = "../data/test-id.RData")

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)
train_id <- train_labels %>% pull(.id)
save(train_id, file = "../data/train-id.RData")

# PCA/projection
proj_out <- projection_fn(train_dtm, 0.7)
train_dtm_projected <- proj_out$data

# Fit Regression
train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

# store predictors and response as matrix and vector
x_train <- train %>% select(-bclass) %>% as.matrix()
y_train <- train_labels %>% pull(bclass)

# fit enet model
alpha_enet <- 0.3
fit_reg <- glmnet(x = x_train, 
                  y = y_train, 
                  family = 'binomial',
                  alpha = alpha_enet)

# choose a constraint strength by cross-validation
cvout <- cv.glmnet(x = x_train, 
                   y = y_train, 
                   family = 'binomial',
                   alpha = alpha_enet)

# store optimal strength
lambda_opt <- cvout$lambda.min

# get log-odds
log_odds <- predict(fit_reg,
                    s = lambda_opt,
                    newx = x_train,
                    type = "link")

save(log_odds, file = "../data/log-odds.RData")

# project test data onto PCs
test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

# coerce to matrix
x_test <- as.matrix(test_dtm_projected)

# compute predicted probabilities
preds <- predict(fit_reg, 
                 s = lambda_opt, 
                 newx = x_test,
                 type = 'response')

# store predictions in a data frame with true labels
pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                              labels = levels(bclass)))

# define classification metric panel 
panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy
pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')