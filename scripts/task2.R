library(tidyverse)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(sparsesvd)
library(glmnet)
library(Matrix)

source("../scripts/modified-preprocessing.R")
source("../scripts/projection.R")

# load raw data
load('../data/claims-raw.RData')

# clean raw data
claims_clean <- claims_raw %>%
  parse_data()

# get bigrams dtm with labels
claims_bigrams <- claims_clean %>% 
  nlp_bg_fn()

save(claims_bigrams, file = '../data/claims-bigrams.RData')
rm(claims_bigrams)

# load id labels for training and test sets
load("../data/train-id.RData")
load("../data/test-id.RData")

# partition claims bigrams based on task1 partition
load("../data/claims-bigrams.RData")
training_bigrams <- claims_bigrams %>%
  filter(
    .id %in% train_id
  )
testing_bigrams <- claims_bigrams %>%
  filter(
    .id %in% test_id
  )
save(training_bigrams, file = '../data/training-bigrams.RData')
save(testing_bigrams, file = '../data/testing-bigrams.RData')
rm(training_bigrams)
rm(testing_bigrams)

# separate DTM from labels
load('../data/testing-bigrams.RData')
test_bg_dtm <- testing_bigrams %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
test_labels <- testing_bigrams %>%
  select(.id, bclass, mclass)
save(test_bg_dtm, file = '../data/test-bg-dtm.RData')
rm(test_bg_dtm)
rm(testing_bigrams)

# same, training set
load('../data/training-bigrams.RData')
train_bg_dtm <- training_bigrams %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
train_labels <- training_bigrams %>%
  select(.id, bclass, mclass)
save(train_bg_dtm, file = '../data/train-bg-dtm.RData')
rm(train_bg_dtm)
rm(training_bigrams)

# PCA/projection
load('../data/train-bg-dtm.RData')
proj_out_bg <- projection_fn(train_bg_dtm, 0.85)
train_bg_dtm_projected <- proj_out_bg$data
rm(train_bg_dtm)

# load log odds from binary logistic principal components regression of word-tokenized data
load("../data/train-log-odds.RData")

# Fit Regression
train_bg <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_bg_dtm_projected) %>%
  bind_cols(train_log_odds)

# store predictors and response as matrix and vector
x_train_bg <- train_bg %>% select(-bclass) %>% as.matrix()
y_train_bg <- train_labels %>% pull(bclass)

# fit enet model
alpha_enet <- 0.3
fit_reg_bg <- glmnet(x = x_train_bg, 
                     y = y_train_bg, 
                     family = 'binomial',
                     alpha = alpha_enet)

# choose a constraint strength by cross-validation
cvout_bg <- cv.glmnet(x = x_train_bg, 
                      y = y_train_bg, 
                      family = 'binomial',
                      alpha = alpha_enet)

# store optimal strength
lambda_opt_bg <- cvout_bg$lambda.min

# project test data onto PCs
load('../data/test-bg-dtm.RData')
test_bg_dtm_projected <- reproject_fn(.dtm = test_bg_dtm, proj_out_bg)
rm(test_bg_dtm)

# bind log-odds and coerce to matrix
load('../data/test-log-odds.RData')
x_test_bg <- test_bg_dtm_projected %>%
  bind_cols(test_log_odds) %>%
  as.matrix()

# compute predicted probabilities
preds <- predict(fit_reg_bg, 
                 s = lambda_opt_bg, 
                 newx = x_test_bg,
                 type = 'response')

# store predictions in a data frame with true labels
pred_df_bg <- test_labels %>%
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
pred_df_bg %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')
