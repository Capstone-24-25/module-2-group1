library(tokenizers)
library(tidyverse)
library(tidytext)
library(dplyr)
library(tm)
library(Matrix)
library(irlba)
source('scripts/preprocessing.R')

# secondary tokenization to get bigrams
bigrams <- claims_raw %>%
  parse_data() %>%
  pull(text_clean) %>%
  tokenize_ngrams(n = 2)

## run PCA on original tokenization (1 word = 1 token) and fit 
## logistic regression on those components -- returns log odds ratio
tokens <- claims_raw %>%
  parse_data() %>%
  pull(text_clean) %>%
  tokenize_words()

tokens <- claims_raw %>%
  parse_data() %>%
  select(.id, text_clean) %>%
  unnest_tokens(word, text_clean)

# create document term matrix
dtm <- tokens %>%
  count(.id, word) %>%
  cast_dtm(document = .id, term = word, value = n)
dtm_matrix <- as.matrix(dtm)

# run PCA
pca_result <- prcomp(dtm_matrix, center = TRUE, scale. = TRUE)

pca_components <- as.data.frame(pca_result$x[, 1:10])
pca_components$.id <- rownames(pca_result$x)

# PCA components to run for log. reg.
response <- claims_raw %>% select(.id, bclass)
data_for_logistic <- pca_components %>%
  left_join(response, by = ".id")

# fit log. reg.
log_reg <- glm(bclass ~ ., data = data_for_logistic %>% select(-.id), family = binomial)
summary(log_reg)

# save log odds
log_odds <- predict(log_reg, type = "link")


## run PCA on bigram tokenization
## create second log. reg. that uses bigram principal components and log odds we just got
bigrams <- claims_raw %>%
  parse_data %>%
  select(.id, text_clean) %>%
  unnest_tokens(bigram, text_clean, token = "ngrams", n = 2)

# create bigram dtm
bigram_dtm <- bigrams %>%
  count(.id, bigram) %>%
  cast_dtm(document = .id, term = bigram, value = n)
bigram_matrix_sparse <- as.matrix(bigram_dtm, 'CsparseMatrix')

# run pca on bigram dtm
bigram_pca <- irlba(bigram_matrix_sparse, nv = 5) # 5 components cuz it wont run with all
bigram_pca_data <- as.data.frame(bigram_pca$u)

# combo log odds from 1st model w 2nd model PC
combined_data <- cbind(bigram_pca_data, as.vector(log_odds))

combined_data <- cbind(bigram_pca_data[order(bigram_pca_data$.id), ], 
                       log_odds[order(log_odds$.id), ]) ## where i stopped

# fit 2nd log reg
bigram_log_reg <- glm(bclass ~ ., data = combined_data, family = binomial)
summary(bigram_log_reg)


## compare two models -- which performs better?

# single tokenization model:
preds_first_model <- predict(log_reg, newdata = claims_test)
pred_class_first_model <- ifelse(preds_first_model > 0.5, 1, 0)

#accuracy1 <- mean(pred_class_first_model == claims_test$bclass)

roc1 <- roc(claims_test$bclass, preds_first_model)
auc1 <- auc(roc1)

# bigram tokenization model:
preds_second_model <- predict(bigram_log_reg, newdata = claims_test)
pred_class_second_model <- ifelse(preds_second_model > 0.5, 1, 0)

#accuracy2 <- mean(pred_class_second_model == claims_test$bclass)

roc2 <- roc(claims_test$bclass, preds_second_model)
auc2 <- auc(roc2)













