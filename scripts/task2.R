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

nlp_bg_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = bigram, 
                  input = text_clean, 
                  token = 'ngrams',
                  n = 2,
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(bigram.lem = lemmatize_strings(bigram)) %>%
    filter(str_length(bigram.lem) > 2) %>%
    count(.id, bclass, mclass, bigram.lem, name = 'n') %>%
    bind_tf_idf(term = bigram.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass', 'mclass'),
                names_from = 'bigram.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

# get bigrams dtm with labels
claims_bigrams <- claims_clean %>% 
  nlp_fn()

save(claims_bigrams, file = '../data/claims-bigrams.RData')

# load log odds from binary logistic principal components regression of word-tokenized data
load("../data/log-odds.RData")

# load id labels for training and test sets
load("../data/train-id.RData")
load("../data/test-id.RData")

# partition claims bigrams
training_bigrams <- claims %>%
  filter(
    .id %in% train_id
  )
testing_bigrams <- claims %>%
  filter(
    .id %in% test_id
  )

# separate DTM from labels
test_bg_dtm <- testing_bigrams %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
test_labels <- training_bigrams %>%
  select(.id, bclass, mclass)

# same, training set
train_bg_dtm <- training_bigrams %>%
  select(-.id, -bclass, -mclass) %>%
  as.matrix()
train_labels <- training_bigrams %>%
  select(.id, bclass, mclass)

# PCA/projection
proj_out_bg <- projection_fn(train_bg_dtm, 0.7)
train_bg_dtm_projected <- proj_out_bg$data

