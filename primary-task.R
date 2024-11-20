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
library(e1071)

source("scripts/modified-preprocessing.R")
source("scripts/projection.R")

# load raw data
load('data/claims-raw.RData')

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
save(test_id, file = "data/test-id.RData")

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)
train_id <- train_labels %>% pull(.id)
save(train_id, file = "data/train-id.RData")

# PCA/projection
proj_out <- projection_fn(train_dtm, 0.7)
train_dtm_projected <- proj_out$data

# Fit SVM
training <- cbind(train_labels %>% select(mclass), train_dtm_projected) %>% as.data.frame()
training %>% names() %>% head(n = 10)
fit_svm <- svm(
  mclass ~ .,
  data = training,
  kernel = "linear",
  cost = 3
)

tune_svm <- tune(
  svm,
  bclass ~ .,
  data = training,
  ranges = list(gamma = 2^(-3:-1), cost = seq(14,24,2))
)

summary(tune_svm)

# project test data onto PCs
test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)
x_test <- test_dtm_projected %>% as.data.frame()

# compute predicted probabilities
preds <- predict(fit_svm, 
                 x_test)

# store predictions in a data frame with true labels
pred_df <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(bclass.pred = preds %>% unname())

pred_df <- test_labels %>%
  transmute(mclass = factor(mclass)) %>%
  bind_cols(mclass.pred = preds %>% unname())

# define classification metric panel 
panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy)

# compute test set accuracy
pred_df %>% panel(truth = bclass, 
                  estimate = bclass.pred,
                  event_level = 'second')

pred_df %>% panel(truth = mclass,
                  estimate = mclass.pred)