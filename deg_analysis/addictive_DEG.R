library(limma)

# Read mapped log count matrix CSV
mapped_log_count_matrix <- read.csv("../data/mapped_log_count_matrix.csv", header = TRUE)

# Remove duplicate rows based on gene names (assuming gene names are in the first column)
mapped_log_count_matrix <- mapped_log_count_matrix[!duplicated(mapped_log_count_matrix[,1]),]

# Set row.names using the first column (gene names) and remove the first column
rownames(mapped_log_count_matrix) <- mapped_log_count_matrix[,1]
mapped_log_count_matrix <- mapped_log_count_matrix[,-1]

# Replace each '.' in the column names with '-'
colnames(mapped_log_count_matrix) <- gsub("\\.", "-", colnames(mapped_log_count_matrix))

# Read meta_info CSV
meta_info <- read.csv("/restricted/projectnb/cte/Challenge_Project_2022/yczhang/data/all_cte_meta_nmf_cleaned.csv", header = TRUE)

# Filter meta_info to keep only rows with sample names in log_count_matrix
sample_names <- colnames(mapped_log_count_matrix)
meta_info_filtered <- meta_info[meta_info$Core_ID %in% sample_names,]

# subset meta_info_filtered based on AgeAtDeath & RIN
meta_info_filtered_subset <- meta_info_filtered[!is.na(meta_info_filtered$RIN) &
                                                  !is.na(meta_info_filtered$AGE), ]
# subset the meta_info_filtered_subset only for European ancestry, race == 1, or AA, race == 2
meta_info_filtered_subset <- meta_info_filtered_subset[grepl(1, meta_info_filtered_subset$race) | grepl(2, meta_info_filtered_subset$race), ]

# subset meta_info_filtered based on H1B1G1 status
meta_info_filtered_subset <- meta_info_filtered_subset[!is.na(meta_info_filtered_subset$rs6910507), ]

# subset meta_info_filtered based on subgroups
meta_info_filtered_subset <- meta_info_filtered_subset[grepl(2, meta_info_filtered_subset$nmf_subtypes), ]

# subset meta_info_filtered_subset based on CTE status
#meta_info_filtered_subset <- meta_info_filtered_subset[!grepl(0, meta_info_filtered_subset$CTE_STAGE) & !is.na(meta_info_filtered_subset$CTE_STAGE), ]

# Subset the mapped_log_count_matrix based on the samples in meta_info_filtered_subset
subset_sample_names <- meta_info_filtered_subset$Core_ID
mapped_log_count_matrix_subset <- mapped_log_count_matrix[, colnames(mapped_log_count_matrix) %in% subset_sample_names]

# Reorder the columns of mapped_log_count_matrix_subset based on the order of sample_conditions
column_order <- match(meta_info_filtered_subset$Core_ID, colnames(mapped_log_count_matrix_subset))
mapped_log_count_matrix_subset <- mapped_log_count_matrix_subset[, column_order]

# create a new design matrix that includes the sample_conditions, RIN, and age_at_death
model_data <- data.frame(
  condition = meta_info_filtered_subset$rs6910507,
  RIN = meta_info_filtered_subset$RIN,
  age_at_death = meta_info_filtered_subset$AGE
)

design_matrix <- model.matrix(~  0 + condition + RIN + age_at_death, data = model_data)

# Fit the model
fit <- lmFit(mapped_log_count_matrix_subset, design_matrix)

# Perform the differential gene expression analysis
fit <- eBayes(fit)
deg_results <- topTable(fit, number = nrow(mapped_log_count_matrix_subset), adjust.method="BH")

# Filter the output to display rows with FDR <= 0.05
filtered_results <- deg_results[deg_results$adj.P.Val <= 0.05, ]
# Save the results to a new CSV file
write.csv(filtered_results, "../data/DEG_results_EA+AA_RIN+AAD_rs6910507_subtype2_.csv", row.names = TRUE)
