if (!require("ggplot2")) install.packages("ggplot2")
if (!require("jsonlite")) install.packages("jsonlite")
if (!require("reshape2")) install.packages("reshape2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("corrplot")) install.packages("corrplot")
if (!require("scales")) install.packages("scales")      

library(ggplot2)
library(jsonlite)
library(reshape2)
library(dplyr)
library(corrplot)
library(scales)

# --- Enhanced Severity Distribution (Log Scale with Percentages) ---
plot_enhanced_severity <- function(data) {
  df_no <- as.data.frame(data$metrics_no_imputation$severity_distribution)
  df_no$Dataset <- "Raw Data"
  
  df_with <- as.data.frame(data$metrics_with_imputation$severity_distribution)
  df_with$Dataset <- "Imputed Data"
  
  combined <- rbind(df_no, df_with)
  
  ggplot(combined, aes(x = factor(Severity), y = count, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), color = "white") +
    scale_y_log10(labels = label_comma()) +
    scale_fill_manual(values = c("#2c3e50", "#27ae60")) +
    theme_minimal(base_size = 14) +
    labs(title = "Impact of Imputation on Severity Class",
         subtitle = "Visualized on Log10 scale to inspect minority classes",
         x = "Severity Level", y = "Accident Count (Log Scale)") +
    theme(legend.position = "top", panel.grid.minor = element_blank())
}

# --- Enhanced Temporal Trends (Hourly + Weekly) ---
plot_temporal_trends <- function(data) {
  # Hourly
  h_no <- as.data.frame(data$metrics_no_imputation$temporal_distributions$hourly) %>% mutate(Dataset = "Raw")
  h_with <- as.data.frame(data$metrics_with_imputation$temporal_distributions$hourly) %>% mutate(Dataset = "Imputed")
  h_combined <- rbind(h_no, h_with)
  
  p1 <- ggplot(h_combined, aes(x = hour, y = count, color = Dataset)) +
    geom_line(size = 1.2, alpha = 0.8) +
    geom_point() +
    scale_x_continuous(breaks = seq(0, 23, 2)) +
    scale_y_continuous(labels = label_comma()) +
    scale_color_manual(values = c("#e67e22", "#34495e")) +
    theme_minimal() +
    labs(title = "Hourly Distribution", x = "Hour of Day", y = "Count")
  
  # Weekly
  day_order <- c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
  w_no <- as.data.frame(data$metrics_no_imputation$temporal_distributions$daily_seasonal) %>% mutate(Dataset = "Raw")
  w_with <- as.data.frame(data$metrics_with_imputation$temporal_distributions$daily_seasonal) %>% mutate(Dataset = "Imputed")
  w_combined <- rbind(w_no, w_with)
  w_combined$weekday <- factor(w_combined$weekday, levels = day_order)
  
  p2 <- ggplot(w_combined, aes(x = weekday, y = count, fill = Dataset)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = c("#e67e22", "#34495e")) +
    theme_minimal() +
    labs(title = "Weekly Distribution", x = "Day", y = "")
  
  return(p1 + p2 + plot_layout(guides = "collect") & theme(legend.position = 'bottom'))
}

# --- Correlation Shift Heatmap ---
plot_correlation_delta <- function(data) {
  # Helper to clean matrix
  get_matrix <- function(m) {
    df <- as.data.frame(m)
    rownames(df) <- df$`_row`
    df$`_row` <- NULL
    return(as.matrix(df))
  }
  
  m_no <- get_matrix(data$metrics_no_imputation$correlation_matrix)
  m_with <- get_matrix(data$metrics_with_imputation$correlation_matrix)
  
  # Calculate Difference
  m_diff <- m_with - m_no
  melted_diff <- melt(m_diff)
  
  ggplot(melted_diff, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.4f", value)), size = 3) +
    scale_fill_gradient2(low = "#d01c8b", mid = "white", high = "#4dac26", midpoint = 0) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Correlation Shift (After - Before Imputation)",
         subtitle = "Green indicates stronger positive correlation after imputation",
         x = "", y = "", fill = "Delta")
}

# --- Categorical Stability Plot ---
plot_cat_sensitivity <- function(data) {
  cat_sens <- as.data.frame(data$sensitivity_comparison$categorical)
  
  ggplot(cat_sens, aes(x = reorder(Variable, Abs_Diff), y = Abs_Diff, fill = Variable)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    guides(fill = "none") +
    theme_minimal() +
    labs(title = "Categorical Feature Stability",
         subtitle = "Maximum Absolute Difference in Proportions",
         x = "Feature", y = "Max Proportion Shift")
}

# --- Stability Combination Plot ---
plot_combined_sensitivity <- function(data, output_path) {
  
  # convert % change to a decimal/fraction to better align with proportion shifts
  sens_num <- as.data.frame(data$sensitivity_comparison$numeric) %>%
    mutate(
      Type = "Numeric (Mean % Change)",
      Impact = Pct_Change / 100, # Scale down to compare with proportions
      Direction = ifelse(Impact >= 0, "Increase/Shift", "Decrease")
    ) %>%
    select(Variable, Impact, Type, Direction)
  
  # take the Absolute Difference (the 'Shift')
  sens_cat <- as.data.frame(data$sensitivity_comparison$categorical) %>%
    group_by(Variable) %>%
    summarise(Impact = max(Abs_Diff)) %>% # Take the most impacted category per variable
    mutate(
      Type = "Categorical (Max Prop. Shift)",
      Direction = "Increase/Shift"
    ) %>%
    select(Variable, Impact, Type, Direction)
  
  combined_sens <- bind_rows(sens_num, sens_cat)
  
  # Create the Visualization
  p <- ggplot(combined_sens, aes(x = reorder(Variable, abs(Impact)), y = Impact, fill = Direction)) +
    geom_col(width = 0.7) +
    # Separate the two types of math into distinct panels
    facet_wrap(~Type, scales = "free_x") + 
    coord_flip() +
    scale_fill_manual(values = c("Decrease" = "#E46666", "Increase/Shift" = "#31a354")) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "#f0f0f0", color = NA),
      strip.text = element_text(face = "bold")
    ) +
    labs(
      title = "Global Sensitivity Analysis",
      subtitle = "Comparing Numeric Mean Shifts vs. Categorical Proportion Stability",
      x = "Feature Name",
      y = "Impact Magnitude (Normalized)",
      fill = "Change Direction"
    )
  
  ggsave(output_path, plot = p, width = 11, height = 7)
}

plot_environmental_tail <- function(data) {
  raw_data <- data
  
  # This targets the post-imputation data used in the clustering
  weather_df <- as.data.frame(raw_data$metrics_with_imputation$weather_distribution)
  
  # Prepare data: Sort and take the top 30 signatures to show the decay
  plot_data <- weather_df %>%
    arrange(desc(count)) %>%
    slice(1:30)
  
  # Define 'Perfect Storm' signatures for highlighting
  high_risk_signatures <- c("Fog", "Heavy Rain", "T-Storm", "Thunder", "Rain")
  
  # Generate the Plot
  ggplot(plot_data, aes(x = reorder(Weather_Condition, count), y = count)) +
    geom_bar(stat = "identity", fill = "#2c3e50", alpha = 0.8) +
    # Highlight the high-risk 'long tail' signatures in red
    geom_bar(data = subset(plot_data, Weather_Condition %in% high_risk_signatures), 
             stat = "identity", fill = "#c0392b") +
    coord_flip() +
    # Use Log10 scale to prove the variance exists across orders of magnitude
    scale_y_log10(labels = comma, expand = expansion(mult = c(0, .1))) +
    labs(
      title = "Weather Distribution",
      subtitle = "Frequency of Incidents by Weather Condition (Log10 Scale)",
      x = "Weather Condition Signature",
      y = "Incident Count (Log Scale)",
      caption = "Red bars highlight low-frequency, high-variance signatures."
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.y = element_text(size = 10),
      panel.grid.minor = element_blank()
    )
}

plot_correlation_severity_focused <- function(data) {
  cor_df <- data$metrics_with_imputation$correlation_matrix
  cor_matrix <- as.matrix(cor_df)
  severity_corrs <- cor_matrix["Severity", ]
  top_features <- names(sort(abs(severity_corrs), decreasing = TRUE)[1:6])
  small_cor_matrix <- cor_matrix[top_features, top_features]
  
  clean_names <- colnames(small_cor_matrix) %>%
    gsub("Weather_Condition_", "", .) %>%
    gsub("Wind_Direction_", "Wind:", .) %>%
    gsub("Temperature\\(F\\)", "Temp", .) %>%
    gsub("Visibility\\(mi\\)", "Vis", .) %>%
    gsub("Humidity\\(%\\)", "Humid", .)
  
  colnames(small_cor_matrix) <- clean_names
  rownames(small_cor_matrix) <- clean_names
  
  # 4. Standardized Plotting 
  # Using method = "number" can sometimes be cleaner if "color" is too busy
  col_palette <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
  
  # Open the PDF device FIRST with a specific size to lock in the scale
  pdf_path <- "./Output/Graphs/EDA/Linear_Correlation.pdf"
  if(!dir.exists(dirname(pdf_path))) dir.create(dirname(pdf_path), recursive = TRUE)
  
  pdf(file = pdf_path, width = 7, height = 7) # Increased to 7x7 for breathing room
  
  corrplot(small_cor_matrix, 
           method = "color", 
           col = col_palette(200),
           type = "upper", 
           addCoef.col = "black",   # The correlation numbers
           
           # --- TEXT SCALING SETTINGS ---
           number.cex = 0.8,        # Size of numbers inside boxes (Reduced to stop overlap)
           tl.cex = 0.8,            # Size of axis labels
           tl.col = "black",        # Label color
           tl.srt = 45,             # Label rotation
           # -----------------------------
           
           diag = FALSE,            
           title = "Top 6 Feature Correlation",
           mar = c(0,0,2,0))
  
  dev.off() # Close the file
  
  message("Plot saved to: ", pdf_path)
}

# --- Execution and Saving ---

eda_data <- fromJSON("./Output/Stats/EDA/eda_comprehensive_results.json")

#ggsave("./Output/Graphs/EDA/severity_detailed.pdf", plot_enhanced_severity(eda_data), width = 6, height = 6)
ggsave("./Output/Graphs/EDA/temporal_analysis.pdf", plot_temporal_trends(eda_data), width = 4, height = 6)
# broken -> ggsave("./Output/Graphs/EDA/correlation_shift.pdf", plot_correlation_delta(eda_data), width = 6, height = 8)
#ggsave("./Output/Graphs/EDA/categorical_sensitivity.pdf", plot_cat_sensitivity(eda_data), width = 6, height = 5)
# broken -> plot_combined_sensitivity(eda_data, "./Output/Graphs/EDA/combined_sensitivity_analysis.pdf")
#ggsave("./Output/Graphs/EDA/categorical_distribution_weather.pdf", plot_environmental_tail(eda_data), width = 6, height = 5)
#plot_correlation_severity_focused(eda_data)

message("Enhanced EDA Visualizations generated successfully.")