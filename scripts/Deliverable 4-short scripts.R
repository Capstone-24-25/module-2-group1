# Load required libraries
library(tidytext)
library(tm)
library(e1071)

# Load saved models
binary_model <- readRDS("binary_model.rds")       # Logistic Regression model
svm_multiclass_model <- readRDS("svm_multiclass_model.rds")  # SVM model

# Load test data
load("data2/claims-test.RData")

# Define function to extract content
extract_content <- function(html_content) {
  tryCatch({
    page <- read_html(html_content)
    headers <- page %>% html_nodes("h1, h2, h3") %>% html_text()
    paragraphs <- page %>% html_nodes("p") %>% html_text()
    combined_content <- c(headers, paragraphs)
    return(paste(combined_content, collapse = " "))
  }, error = function(e) {
    return(NA)
  })
}

# Step 1: Preprocess test data
claims_test$content_combined <- sapply(claims_test$text_tmp, extract_content)

# Tokenize test data
test_tokens <- claims_test %>%
  unnest_tokens(word, content_combined) %>%
  anti_join(stop_words, by = "word") %>%
  count(.id, word)

# Create Document-Term Matrix (DTM) for test data
test_dtm <- test_tokens %>%
  cast_dtm(document = .id, term = word, value = n)

# Align DTM with the training DTM
load("train_dtm.rds")  # Load training DTM for alignment
missing_cols <- setdiff(colnames(train_dtm), colnames(test_dtm))
test_dtm_aligned <- cbind(test_dtm, matrix(0, nrow = nrow(test_dtm), ncol = length(missing_cols)))
colnames(test_dtm_aligned) <- c(colnames(test_dtm), missing_cols)
test_dtm_aligned <- test_dtm_aligned[, colnames(train_dtm), drop = FALSE]
test_matrix <- as.matrix(test_dtm_aligned)

# Step 2: Apply PCA transformation
load("pca_result.rds")  # Load pre-trained PCA model
test_pca_scores <- predict(pca_result, newdata = test_matrix)

# Reduce PCA components to match training
test_pca_reduced <- test_pca_scores[, 1:(ncol(binary_model$coefficients) - 1)]

# Step 3: Generate predictions

# Predict binary labels
test_pca_df <- as.data.frame(test_pca_reduced)
binary_preds <- predict(binary_model, newdata = test_pca_df, type = "response")
binary_labels <- ifelse(binary_preds > 0.5, "1", "0")

# Predict multiclass labels
multiclass_preds <- predict(svm_multiclass_model, test_pca_reduced)

# Combine predictions into final data frame
pred_df <- data.frame(
  .id = claims_test$.id,
  bclass.pred = binary_labels,
  mclass.pred = multiclass_preds
)

# Save predictions
saveRDS(pred_df, "predictions.rds")
