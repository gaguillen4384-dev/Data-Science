library(data.table) 
library(ggplot2)
library(jsonlite)
library(sf)

# --- METRIC FUNCTIONS  ---

# Crash Severity Index (CSI)
calculate_metrics_csi <- function(dt) {
  dt[, weight := fifelse(Severity == 4, 5, 
                         fifelse(Severity == 3, 3, 
                                 fifelse(Severity == 2, 2, 1)))]
  
  summary <- dt[, .(
    Total_Crashes = .N,
    Total_CSI_Score = sum(weight),
    Avg_CSI_per_Crash = mean(weight)
  )]
  return(as.list(summary))
}

# Infrastructure Presence Ratio
calculate_metrics_infra <- function(dt) {
  infra_cols <- c("Amenity", "Bump", "Crossing", "Give_Way", "Junction", 
                  "No_Exit", "Railway", "Roundabout", "Station", "Stop", 
                  "Traffic_Calming", "Traffic_Signal")
  
  # Since data is 0/1, the mean of the column is the percentage of presence.
  res <- dt[, lapply(.SD, mean, na.rm = TRUE), .SDcols = infra_cols]
  return(as.list(res))
}

# Infrastructure Distribution Metrics
calculate_metrics_binary <- function(dt) {
  infra_cols <- c("Amenity", "Bump", "Crossing", "Give_Way", "Junction", 
                  "No_Exit", "Railway", "Roundabout", "Station", "Stop", 
                  "Traffic_Calming", "Traffic_Signal")
  
  # Melt the data.table efficiently
  long_dt <- melt(dt, measure.vars = infra_cols, variable.name = "Infra_Feature")
  
  # Filter where value is 1 (Active infrastructure)
  binary_dist <- long_dt[value == 1, .N, by = Infra_Feature]
  
  return(binary_dist)
}
# --- VISUALIZATION FUNCTIONS ---

plot_kde <- function(dt) {
  p <- ggplot(dt, aes(x = Start_Lng, y = Start_Lat)) +
    stat_bin_2d(bins = 100) + 
    scale_fill_viridis_c(option = "magma") +
    theme_minimal() +
    labs(
      title = "Spatial Crash Hotspots",
      subtitle = "Binned density records",
      x = "Longitude",
      y = "Latitude",
      fill = "Crash Count"
    )
  
  # Explicitly print to the PDF device
  pdf("./Output/Graphs/EDA/plot_spatial_density.pdf", width = 8, height = 8)
  print(p) 
  invisible(dev.off())
}

plot_cross_tab <- function(dt) {
  infra_cols <- c("Amenity", "Bump", "Crossing", "Give_Way", "Junction", 
                  "No_Exit", "Railway", "Roundabout", "Station", "Stop", 
                  "Traffic_Calming", "Traffic_Signal")
  
  # Melt the data for large rows count
  agg_dt <- melt(dt, id.vars = "Severity", measure.vars = infra_cols)

  # use value == 1 or as.logical(value) to capture binary markers
  agg_dt <- agg_dt[value == 1, .N, by = .(Severity, variable)]
  
  if (nrow(agg_dt) == 0) {
    message("Warning: No infrastructure markers (value == 1) found. Check your column data types.")
    return(NULL)
  }
  
  # Generate the Heatmap
  p <- ggplot(agg_dt, aes(x = variable, y = factor(Severity), fill = N)) +
    geom_tile() +
    # Changed low to 'gray95' so the tiles are visible even with low counts
    scale_fill_gradient(low = "gray95", high = "red") + 
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Severity vs Infrastructure (Heatmap)",
         x = "Infrastructure Type",
         y = "Severity Level",
         fill = "Accident Count")
  
  pdf("./Output/Graphs/EDA/plot_severity_infrastructure.pdf", width = 8, height = 8)
  print(p)
  invisible(dev.off())
}

plot_temporal <- function(dt) {
  # Aggregate and plot hourly data 
  hour_agg <- dt[, .N, by = .(Hour, Severity)]
  p1 <- ggplot(hour_agg, aes(x = Hour, y = N, fill = factor(Severity))) +
    geom_bar(stat = "identity") + 
    scale_fill_viridis_d() + # Different color scale for distinction
    theme_minimal() +
    labs(title = "Crashes by Hour (Stacked by Severity)", fill = "Severity")
  
  pdf("./Output/Graphs/EDA/plot_temporal_hour.pdf", width = 8, height = 8)
  print(p1)
  invisible(dev.off())
  
  # Aggregate and plot day data
  day_agg <- dt[, .N, by = .(DayOfWeek, Severity)]
  
  # IMPORTANT: Map labels to match your wday() settings.
  # If Sunday = 1 (default wday):
  day_labels <- c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")
  
  # If used week_start = 1 (Monday = 1), use this instead:
  # day_labels <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
  
  # Apply the factor levels to prevent the 'One Big Chunk' error
  day_agg[, DayOfWeek := factor(DayOfWeek, levels = 1:7, labels = day_labels)]
  
  p2 <- ggplot(day_agg, aes(x = DayOfWeek, y = N, fill = factor(Severity))) +
    geom_bar(stat = "identity") + # Stacked by default
    scale_fill_brewer(palette = "Set1") + 
    theme_minimal() +
    labs(title = "Crashes by Day of Week (Stacked by Severity)", 
         x = "Day of Week", 
         y = "Total Accidents",
         fill = "Severity")
  
  pdf("./Output/Graphs/EDA/plot_temporal_day.pdf", width = 8, height = 8)
  print(p2)
  invisible(dev.off())
}

# --- EXECUTION FLOW ---
message("Execution Flow Started")
dt <- fread("./Datasets/Preprocessing/clean_enriched_cfl.csv")

# Compile Metrics to ONE JSON
all_metrics <- list(
  csi_summary = calculate_metrics_csi(dt),
  infra_ratios = calculate_metrics_infra(dt),
  binary_counts = calculate_metrics_binary(dt)
)
write_json(all_metrics, "./Output/Stats/EDA/crash_analysis_metrics.json", pretty = TRUE)

# Generate SEPARATE PDFs
plot_kde(dt)
plot_cross_tab(dt)
plot_temporal(dt)
message("Execution Flow Completed")
