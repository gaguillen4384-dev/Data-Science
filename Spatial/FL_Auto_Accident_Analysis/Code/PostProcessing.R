library(tidyverse)
library(sf)
library(ggspatial)
library(jsonlite)

# --- Functions ---

# Load and Merge Data
prepare_cluster_data <- function(raw_path, mapping_path) {
  raw_df <- read.csv(raw_path)
  mapping_df <- read.csv(mapping_path)
  
  # Join and filter out noise (Cluster 0)
  # Note: Metrics snoise pointsto be excluded
  merged_df <- raw_df %>%
    inner_join(mapping_df, by = "ID") %>%
    filter(Cluster != 0)
  
  return(merged_df)
}

# Get Top n Clusters
get_top_clusters <- function(df, n = 5, weight_freq = 0.4, weight_sev = 0.6, json_output_path = "./Output/Stats/PostProcessing/top_priority_clusters.json") {
  # This looks for any column containing the word "Severity" (case insensitive)
  sev_col <- grep("Severity", names(df), value = TRUE, ignore.case = TRUE)[1]
  infra_cols <- c("Traffic_Signal", "Junction", "Crossing", "Station")

  message(paste("Using column '", sev_col, "' for severity calculations...", sep=""))
  
  # Calculate Frequency and Severity per cluster
  
  top_summary <- df %>%
    group_by(Cluster) %>%
    summarise(
      Frequency = n(),
      Total_Infra = sum(across(all_of(infra_cols)), na.rm = TRUE),
      # .data[[sev_col]] tells R to use the column name stored in the variable
      Total_Severity = sum(.data[[sev_col]], na.rm = TRUE), 
      Avg_Severity = mean(.data[[sev_col]], na.rm = TRUE),
      Peak_Hour_Raw = as.numeric(names(sort(table(Hour), decreasing = TRUE)[1])),
      Peak_Hour_Civilian = format(as.POSIXct(paste(Peak_Hour_Raw, ":00", sep=""), format="%H:%M"), "%I %p"),
      Primary_Day = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")[
        as.numeric(names(sort(table(DayOfWeek), decreasing = TRUE)[1]))
      ]
    ) %>%
    # Weighted Score
    mutate(
      Priority_Score = (Frequency * weight_freq) + (Avg_Severity * weight_sev)
    ) %>%
    arrange(desc(Priority_Score)) %>%
    head(n)
  
  # ... [Rest of the function stays the same] ...
  top_ids <- top_summary %>% pull(Cluster)
  
  cluster_labels <- c(
    "228" = "Afternoon Transit Throughway",
    "320" = "Evening Commuter Corridor",
    "137" = "Peak Hour Transit Link",
    "290" = "Late-Day Commute Zone",
    "150" = "Weekend Activity Hub"
  )
  
  json_data <- top_summary %>%
    mutate(
      Label = ifelse(as.character(Cluster) %in% names(cluster_labels), 
                     cluster_labels[as.character(Cluster)], 
                     "Other/Unlabeled")
    )
  
  write_json(json_data, json_output_path, pretty = TRUE)
  message(paste("Hybrid Priority JSON exported to:", json_output_path))
  
  return(df %>% filter(Cluster %in% top_ids))
}

get_top_infra_clusters <- function(df, n = 5, weight_infra = 0.7, weight_sev = 0.4, json_output_path = "./Output/Stats/PostProcessing/top_infra_priority.json") {
  
  # Identify Infrastructure and Severity columns
  infra_cols <- c("Traffic_Signal", "Junction", "Crossing", "Station")
  sev_col <- grep("Severity", names(df), value = TRUE, ignore.case = TRUE)[1]
  
  message(paste("Ranking by Infrastructure influence vs.", sev_col, "..."))
  
  # Calculate scores
  top_summary <- df %>%
    group_by(Cluster) %>%
    summarise(
      # Calculate the mean presence of infrastructure across all points in the cluster
      Infra_Presence = mean(rowSums(across(all_of(infra_cols)), na.rm = TRUE)),
      Total_Infra = sum(across(all_of(infra_cols)), na.rm = TRUE),
      Avg_Severity = mean(.data[[sev_col]], na.rm = TRUE),
      Total_Severity = sum(.data[[sev_col]], na.rm = TRUE),
      Frequency = n(),
      Peak_Hour_Raw = as.numeric(names(sort(table(Hour), decreasing = TRUE)[1])),
      Peak_Hour_Civilian = format(as.POSIXct(paste(Peak_Hour_Raw, ":00", sep=""), format="%H:%M"), "%I %p"),
      Primary_Day = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")[
        as.numeric(names(sort(table(DayOfWeek), decreasing = TRUE)[1]))
      ]
    ) %>%
    # Weighted Score: Normalizing high severity incidents at high-infra locations
    mutate(
      # scale both to 0-1 range (optional but recommended) to ensure weights are meaningful
      Priority_Score = (Total_Infra * weight_infra) + (Avg_Severity * weight_sev)
    ) %>%
    arrange(desc(Priority_Score)) %>%
    head(n)
  
  # Labeling and JSON Export
  cluster_labels <- c(
    "1" = "Evening Commuter Intersection",
    "4" = "Late-Night Signalized Zone",
    "6" = "Early Morning Signalized Crossing",
    "7" = "Morning Multi-Modal Hub",
    "9" = "Mid-Morning Transit Point"
  )
  
  json_data <- top_summary %>%
    mutate(
      Label = ifelse(as.character(Cluster) %in% names(cluster_labels), 
                     cluster_labels[as.character(Cluster)], 
                     "Other/Unlabeled")
    )
  
  write_json(json_data, json_output_path, pretty = TRUE)
  message(paste("Infrastructure Priority JSON exported to:", json_output_path))
  
  top_ids <- top_summary %>% pull(Cluster)
  return(df %>% filter(Cluster %in% top_ids))
}

get_top_severity_clusters <- function(df, n = 5, json_output_path = "./Output/Stats/PostProcessing/top_severity_only.json") {
  
  # Identify the severity column
  sev_col <- grep("Severity", names(df), value = TRUE, ignore.case = TRUE)[1]
  
  message("Filtering for maximum severity (Level 3) only...")
  
  # Filter for high severity and aggregate
  top_summary <- df %>%
    filter(.data[[sev_col]] == 3) %>%
    group_by(Cluster) %>%
    summarise(
      Frequency = n(),
      Avg_Severity = mean(.data[[sev_col]], na.rm = TRUE),
      Peak_Hour_Raw = as.numeric(names(sort(table(Hour), decreasing = TRUE)[1])),
      Peak_Hour_Civilian = format(as.POSIXct(paste(Peak_Hour_Raw, ":00", sep=""), format="%H:%M"), "%I %p"),
      Primary_Day = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")[
        as.numeric(names(sort(table(DayOfWeek), decreasing = TRUE)[1]))
      ]
    ) %>%
    # 2. Sort by Frequency (since Severity is a constant 3)
    arrange(desc(Frequency)) %>%
    head(n)
  
  # Labeling Logic for fatal zones
  cluster_labels <- c(
    "79"  = "Major Fatal Hub",
    "181" = "High-Volume Fatal Corridor",
    "62"  = "Critical Severity Zone",
    "263" = "Priority Safety Sector",
    "57"  = "Extreme Risk Area"
  )
  
  json_data <- top_summary %>%
    mutate(
      Label = ifelse(as.character(Cluster) %in% names(cluster_labels), 
                     cluster_labels[as.character(Cluster)], 
                     "Unlabeled Fatal Cluster")
    )
  
  # Export results
  write_json(json_data, json_output_path, pretty = TRUE)
  message(paste("Pure Severity JSON exported to:", json_output_path))
  
  top_ids <- top_summary %>% pull(Cluster)
  return(df %>% filter(Cluster %in% top_ids))
}

get_top_fatal_infra_clusters <- function(df, n = 5, weight_infra = 0.7, weight_sev = 0.4, json_output_path = "./Output/Stats/PostProcessing/top_severity3_infra.json") {
  
  infra_cols <- c("Traffic_Signal", "Junction", "Crossing", "Station")
  sev_col <- grep("Severity", names(df), value = TRUE, ignore.case = TRUE)[1]
  
  message("Filtering for Severity 3 and ranking by Infrastructure...")
  
  # Filter and Score
  top_summary <- df %>%
    # Only keep the most severe clusters
    filter(.data[[sev_col]] == 3) %>%
    group_by(Cluster) %>%
    summarise(
      Infra_Presence = mean(rowSums(across(all_of(infra_cols)), na.rm = TRUE)),
      Total_Infra = sum(across(all_of(infra_cols)), na.rm = TRUE),
      Avg_Severity = mean(.data[[sev_col]], na.rm = TRUE),
      Total_Severity = sum(.data[[sev_col]], na.rm = TRUE),
      Frequency = n(),
      Peak_Hour_Raw = as.numeric(names(sort(table(Hour), decreasing = TRUE)[1])),
      Peak_Hour_Civilian = format(as.POSIXct(paste(Peak_Hour_Raw, ":00", sep=""), format="%H:%M"), "%I %p"),
      Primary_Day = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")[
        as.numeric(names(sort(table(DayOfWeek), decreasing = TRUE)[1]))
      ]
    ) %>%
    mutate(
      Priority_Score = (Total_Infra * weight_infra) + (Avg_Severity * weight_sev)
    ) %>%
    # Rank and pick top N
    arrange(desc(Priority_Score), desc(Frequency)) %>%
    head(n)
  
  # Labeling Logic
  cluster_labels <- c(
    "79"  = "High-Severity Rural Link",
    "235" = "Fatal Unsignalized Zone",
    "14"  = "Max-Severity Corridor"
  )
  
  json_data <- top_summary %>%
    mutate(
      Label = ifelse(as.character(Cluster) %in% names(cluster_labels), 
                     cluster_labels[as.character(Cluster)], 
                     "High-Severity / Low-Infra")
    )
  
  write_json(json_data, json_output_path, pretty = TRUE)
  message(paste("Fatal Infrastructure JSON exported to:", json_output_path))
  
  return(df %>% filter(Cluster %in% top_summary$Cluster))
}

generate_cluster_profiles <- function(df, output_path = "./Output/Stats/PostProcessing/cluster_profiles.csv") {

  profile <- df %>%
    group_by(Cluster) %>%
    summarise(
      Total_Points = n(),
      # Spatial Center
      Centroid_Lat = mean(Start_Lat, na.rm = TRUE),
      Centroid_Lng = mean(Start_Lng, na.rm = TRUE),
      # Average Severity and Rank
      Avg_Severity = mean(Severity, na.rm = TRUE),
      Avg_Rank = mean(RANK, na.rm = TRUE),
      # Environmental Features (Percentage of points near these features)
      Pct_Traffic_Signal = mean(Traffic_Signal, na.rm = TRUE) * 100,
      Pct_Junction = mean(Junction, na.rm = TRUE) * 100,
      Pct_Crossing = mean(Crossing, na.rm = TRUE) * 100,
      Pct_Station = mean(Station, na.rm = TRUE) * 100,
      # Temporal Patterns
      Peak_Hour = as.numeric(names(sort(table(Hour), decreasing = TRUE)[1])),
      Primary_Day = c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")[
        as.numeric(names(sort(table(DayOfWeek), decreasing = TRUE)[1]))
      ]
    ) %>%
    arrange(desc(Total_Points))
  
  write.csv(profile, output_path, row.names = FALSE)
  return(profile)
}

# --- MAP FUNCTIONS ---

# Generates a more detailed and useful map
plot_academic_detailed_map <- function(cluster_df, top_cluster_number = 5) {
  # grabbed from https://overpass-turbo.eu/
  # search "Orange County, Florida"
  # in the wizard: highway=motorway or highway=primary or highway=secondary or highway=tertiary
  # build and run query -> k Export (top menu) and under the "Data" section, choose download as GeoJSON
  road_data_path <- "./Datasets/orange_county_roads.geojson"
  
  if (!file.exists(road_data_path)) {
    stop("Road data file not found! Please download it from Overpass Turbo.")
  }
  
  # Load and Split Geometry
  raw_osm <- st_read(road_data_path, quiet = TRUE)
  
  # Separate Lines (Roads)
  road_lines <- raw_osm %>% 
    filter(st_geometry_type(.) %in% c("LINESTRING", "MULTILINESTRING"))
  
  # Separate Markers (Points) safely
  road_markers <- raw_osm %>% 
    filter(st_geometry_type(.) == "POINT")
  
  # Check if our target columns exist before filtering to avoid the error
  existing_cols <- colnames(road_markers)
  
  if ("highway" %in% existing_cols | "place" %in% existing_cols) {
    road_markers <- road_markers %>%
      filter(if_any(any_of(c("highway", "place")), ~ !is.na(.)))
  } else {
    # If no tags exist, just take the first 50 points so the map isn't empty, 
    # or leave as is
    message("Note: 'highway' or 'place' tags not found in markers. Showing all points.")
  }
  # Prepare Cluster Data
  points_sf <- st_as_sf(cluster_df, coords = c("Start_Lng", "Start_Lat"), crs = 4326)
  my_colors <- c("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E")
  top_5_labels <- c(
    "228" = "Afternoon Transit Throughway",
    "320" = "Evening Commuter Corridor",
    "137" = "Peak Hour Transit Link",
    "290" = "Late-Day Commute Zone",
    "150" = "Weekend Activity Hub"
  )
  
  # Plotting
  ggplot() +
    # Layer 1: Road Lines
    geom_sf(data = road_lines, color = "grey85", linewidth = 0.2, alpha = 0.8) +
    
    # Layer 2: Infrastructure Markers
    geom_sf(data = road_markers, color = "slateblue", size = 0.8, alpha = 0.4, shape = 18) +
    
    # Layer 3: Your Clusters
    geom_sf(data = points_sf, aes(color = as.factor(Cluster), shape = as.factor(Cluster)),
            size = 1.8, alpha = 0.8) +
    
    # Zoom to Orange County bounds
    coord_sf(xlim = c(-81.85, -80.90), ylim = c(28.25, 28.85), expand = FALSE) +
    
    # --- Unified Scales ---
    scale_color_manual(
      name = "Cluster Hotspots",   
      values = my_colors, 
      labels = top_5_labels        
    ) + 
    scale_shape_manual(
      name = "Cluster Hotspots",   
      values = c(16, 17, 15, 8, 4, 3), 
      labels = top_5_labels        
    ) + 
    
    # Academic Theme
    theme_minimal() +
    theme(
      text = element_text(family = "serif"),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid = element_line(color = "grey95"),
      
      # --- Legend Update ---
      legend.position = "bottom",          # Moves legend to bottom
      legend.direction = "horizontal",    # Forces horizontal layout
      legend.box = "horizontal",          # Ensures combined scales align horizontally
      
      legend.title = element_text(face = "bold"),
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10, color = "grey30")
    ) +
    
    labs(
      title = "DBSCAN Cluster Analysis: Orange County, FL",
      subtitle = paste0("Top ", top_cluster_number, " Frequency Clusters"),
      x = "Longitude", y = "Latitude"
    ) +
    
    # Spatial Elements
    annotation_scale(location = "br", width_hint = 0.4) +
    annotation_north_arrow(location = "tl", style = north_arrow_minimal())
}

# Generate Academic Visualization with Local Shapefile
plot_cluster_map_with_shp <- function(cluster_df, shapefile_path, top_cluster_number = 5) {
  # Load shapefile
  cfl_map <- st_read(shapefile_path)
  
  # Coordinate conversion
  points_sf <- st_as_sf(cluster_df, coords = c("Start_Lng", "Start_Lat"), crs = 4326)
  cfl_map <- st_transform(cfl_map, 4326)
  
  ggplot() +
    # THE "LAND": Set the entire panel to black to give that full-map feel
    # THE LINES: Draw the shapefile lines in a very dark grey or thin white
    geom_sf(data = cfl_map, color = "grey20", size = 0.1) +
    
    # THE DATA: White points with high contrast
    # Using alpha for density and shapes for distinction
    geom_sf(data = points_sf, aes(shape = as.factor(Cluster)), 
            color = "white", size = 1.5, alpha = 0.7) +
    
    # Formatting
    scale_shape_manual(values = c(16, 17, 15, 3, 8)) + 
    theme_minimal() +
    theme(
      # This makes the "empty" parts of your map black instead of white
      panel.background = element_rect(fill = "black"),
      plot.background = element_rect(fill = "white"), # Outer border remains white for the report
      panel.grid = element_blank(),
      text = element_text(family = "serif"),
      legend.position = "right"
    ) +
    labs(
      title = "DBSCAN Cluster Analysis: Orange County Hotspots",
      shape = "Cluster ID",
      x = "Longitude",
      y = "Latitude"
    ) +
    annotation_scale(location = "bl", bar_cols = c("white", "grey")) +
    annotation_north_arrow(location = "tl", style = north_arrow_minimal(line_col = "white", text_col = "white"))
}

plot_downtown_zoom_map <- function(cluster_df, top_cluster_number = 5) {
  # grabbed from https://overpass-turbo.eu/
  # search "Orange County, Florida"
  # in the wizard: highway=motorway or highway=primary or highway=secondary or highway=tertiary or highway=residential
  # build and run query -> k Export (top menu) and under the "Data" section, choose download as GeoJSON
  road_data_path <- "./Datasets/western_orlando_roads.geojson"
  if (!file.exists(road_data_path)) stop("File not found.")
  
  raw_osm <- st_read(road_data_path, quiet = TRUE)
  
  top_5_labels <- c(
        "79"  = "High-Severity Rural Link",
        "181" = "Morning Commute Risk Zone",
        "62"  = "Mid-Week Severity Hotspot",
        "263" = "Early Morning High-Risk",
        "57"  = "Low-Infra Priority Area"
  )
  
  labeled_roads <- raw_osm %>%
    filter(st_geometry_type(.) %in% c("LINESTRING", "MULTILINESTRING")) %>%
    filter(!is.na(name)) %>%
    group_by(name, highway) %>% 
    summarize(geometry = st_union(geometry), .groups = "drop") %>%
    # Only label major-ish roads to keep it clean
    filter(highway %in% c("motorway", "primary", "secondary", "tertiary"))
  
  # The background roads (all of them, but without labels)
  all_roads <- raw_osm %>% 
    filter(st_geometry_type(.) %in% c("LINESTRING", "MULTILINESTRING"))
  
  # Bounding Box
  west_xlim <- c(-81.45, -81.36)
  west_ylim <- c(28.50, 28.56)
  points_sf <- st_as_sf(cluster_df, coords = c("Start_Lng", "Start_Lat"), crs = 4326)
  my_colors <- c("#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E")
  ggplot() +
    # Layer 1 & 2: Roads and Labels
    geom_sf(data = all_roads, color = "grey90", linewidth = 0.2) +
    geom_sf_text(data = labeled_roads, aes(label = name), size = 1.8, 
                 family = "serif", color = "grey50", check_overlap = TRUE) +
    
    # Layer 3: Clusters
    geom_sf(data = points_sf, aes(color = as.factor(Cluster), shape = as.factor(Cluster)),
            size = 2.5, alpha = 0.8) +
    
    coord_sf(xlim = west_xlim, ylim = west_ylim, expand = FALSE, datum = NA) +
    
    scale_color_manual(
      name = "Cluster Locations",
      values = my_colors,
      labels = top_5_labels
    ) +
    scale_shape_manual(
      name = "Cluster Locations",
      values = c(16, 17, 15, 8, 4, 3), # Ensure enough shapes for all clusters
      labels = top_5_labels
    ) +theme_minimal() +
    theme(
      text = element_text(family = "serif"),
      panel.grid = element_blank(),
      axis.title = element_blank(),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.text = element_text(size = 7),
      legend.title = element_text(size = 8, face = "bold"),
      legend.margin = margin(t = -5)
    ) +
    guides(
      color = guide_legend(nrow = 1, byrow = TRUE),
      shape = guide_legend(nrow = 1, byrow = TRUE)
    )+
    labs(title = "Western Orlando: Focused By Frequency Cluster View")
}

plot_high_severity_clusters <- function(cluster_df, top_cluster_number = 5) {
  # Path to the GeoJSON data extracted from Overpass Turbo
  road_data_path <- "./Datasets/western_orlando_roads.geojson"
  if (!file.exists(road_data_path)) stop("File not found.")
  
  # Load road data
  raw_osm <- st_read(road_data_path, quiet = TRUE)
  
  # Updated labels based on your high-severity cluster data
  top_5_labels <- c(
    "228" = "Afternoon Transit Throughway",
    "320" = "Evening Commuter Corridor",
    "137" = "Peak Hour Transit Link",
    "290" = "Late-Day Commute Zone",
    "150" = "Weekend Activity Hub"
  )
  
  # Process roads for labeling (Major arteries only)
  labeled_roads <- raw_osm %>%
    filter(st_geometry_type(.) %in% c("LINESTRING", "MULTILINESTRING")) %>%
    filter(!is.na(name)) %>%
    group_by(name, highway) %>% 
    summarize(geometry = st_union(geometry), .groups = "drop") %>%
    filter(highway %in% c("motorway", "primary", "secondary", "tertiary"))
  
  # Prepare background road layer
  all_roads <- raw_osm %>% 
    filter(st_geometry_type(.) %in% c("LINESTRING", "MULTILINESTRING"))
  
  # Define Spatial Bounding Box (Western Orlando)
  west_xlim <- c(-81.45, -81.36)
  west_ylim <- c(28.50, 28.56)
  
  # Convert cluster data to spatial features
  points_sf <- st_as_sf(cluster_df, coords = c("Start_Lng", "Start_Lat"), crs = 4326)
  
  # Color palette for the 5 clusters
  my_colors <- c("#D73027", "#FC8D59", "#74B652", "#91BFDB", "#7B52AE")
  
  ggplot() +
    # Layer 1: Background Road Network
    geom_sf(data = all_roads, color = "grey92", linewidth = 0.2) +
    
    # Layer 2: Road Name Labels
    geom_sf_text(data = labeled_roads, aes(label = name), size = 1.8, 
                 family = "serif", color = "grey60", check_overlap = TRUE) +
    
    # Layer 3: High-Severity Clusters
    geom_sf(data = points_sf, aes(color = as.factor(Cluster), shape = as.factor(Cluster)),
            size = 3, alpha = 0.9) +
    
    # Set coordinates and crop to area of interest
    coord_sf(xlim = west_xlim, ylim = west_ylim, expand = FALSE, datum = NA) +
    
    # Aesthetic scales
    scale_color_manual(
      name = "High-Severity Clusters",
      values = my_colors,
      labels = top_5_labels
    ) +
    scale_shape_manual(
      name = "High-Severity Clusters",
      values = c(17, 16, 15, 18, 8), 
      labels = top_5_labels
    ) +
    
    # Thematic styling
    theme_minimal() +
    theme(
      text = element_text(family = "serif"),
      panel.grid = element_blank(),
      axis.title = element_blank(),
      legend.position = "bottom",
      legend.box = "vertical",
      legend.text = element_text(size = 7),
      legend.title = element_text(size = 8, face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    guides(
      color = guide_legend(ncol = 2),
      shape = guide_legend(ncol = 2)
    ) +
    labs(title = "Western Orlando: High-Severity Incident Clusters")
}

plot_infrastructure_priority_map <- function(cluster_df, top_cluster_number = 5) {
  # Path to local road data
  road_data_path <- "./Datasets/orange_county_roads.geojson"
  
  if (!file.exists(road_data_path)) {
    stop("Road data file not found! Please ensure orange_county_roads.geojson is in the Datasets folder.")
  }
  
  # Load and Split Geometry
  raw_osm <- st_read(road_data_path, quiet = TRUE)
  
  # Separate Lines (Roads)
  road_lines <- raw_osm %>% 
    filter(st_geometry_type(.) %in% c("LINESTRING", "MULTILINESTRING"))
  
  # Separate Markers (Points) safely
  road_markers <- raw_osm %>% 
    filter(st_geometry_type(.) == "POINT")
  
  existing_cols <- colnames(road_markers)
  if ("highway" %in% existing_cols | "place" %in% existing_cols) {
    road_markers <- road_markers %>%
      filter(if_any(any_of(c("highway", "place")), ~ !is.na(.)))
  }

  top_5_labels <- c(
    "228" = "Afternoon Transit Throughway",
    "320" = "Evening Commuter Corridor",
    "137" = "Peak Hour Transit Link",
    "290" = "Late-Day Commute Zone",
    "150" = "Weekend Activity Hub"
  )
  
  # Prepare Cluster Data
  points_sf <- st_as_sf(cluster_df, coords = c("Start_Lng", "Start_Lat"), crs = 4326)
  
  # Higher contrast palette for academic reporting
  my_colors <- c("#D73027", "#FC8D59", "#74B652", "#91BFDB", "#7B52AE")
  
  # Plotting
  ggplot() +
    # Layer 1: Road Lines
    geom_sf(data = road_lines, color = "grey88", linewidth = 0.15, alpha = 0.7) +
    
    # Layer 2: Infrastructure Markers (OSM Nodes)
    geom_sf(data = road_markers, color = "darkslategrey", size = 0.5, alpha = 0.3, shape = 18) +
    
    # Layer 3: Incident Clusters from top_severity3_infra_2.json[cite: 2]
    geom_sf(data = points_sf, aes(color = as.factor(Cluster), shape = as.factor(Cluster)),
            size = 2.2, alpha = 0.9) +
    
    # County Bounds Zoom
    coord_sf(xlim = c(-81.85, -80.90), ylim = c(28.25, 28.85), expand = FALSE) +
    
    # --- Unified Scales ---
    scale_color_manual(
      name = "Priority Severity Clusters",   
      values = my_colors, 
      labels = top_5_labels        
    ) + 
    scale_shape_manual(
      name = "Priority Severity Clusters",   
      values = c(17, 16, 15, 18, 8), 
      labels = top_5_labels        
    ) + 
    
    # Academic Theme
    theme_minimal() +
    theme(
      text = element_text(family = "serif"),
      panel.background = element_rect(fill = "white", color = NA),
      panel.grid = element_line(color = "grey96"),
      
      legend.position = "bottom",          
      legend.direction = "horizontal",    
      legend.box = "horizontal",          
      legend.text = element_text(size = 7),
      
      legend.title = element_text(face = "bold", size = 9),
      plot.title = element_text(size = 13, face = "bold"),
      plot.subtitle = element_text(size = 9, color = "grey25")
    ) +
    
    labs(
      title = "Infrastructure Priority Analysis: Orange County, FL",
      subtitle = "High-Severity Level 3 Incidents",
      x = "Longitude", y = "Latitude"
    ) +
    
    # Spatial Elements
    annotation_scale(location = "br", width_hint = 0.3) +
    annotation_north_arrow(location = "tl", style = north_arrow_minimal())
}

# --- Execution Flow ---
clean_cfl_path <- "./Datasets/Preprocessing/clean_enriched_cfl.csv"
cluster_mappings_path <- "./Datasets/Model_Engineering/cluster_mapping.csv"
cfl_state_road <- "./Datasets/Central_FL_Roads.shp"
top_cluster_number <- 5

message("Execution Flow Started")
full_data <- prepare_cluster_data(clean_cfl_path, cluster_mappings_path)

# Give meaning to clusters
#cluster_summary <- generate_cluster_profiles(full_data)

# Extract top clusters

#top_clusters <- get_top_infra_clusters(full_data, n = top_cluster_number)
#get_top_severity_clusters(full_data, n = top_cluster_number)

# --- TOP 5 By Frequency ---
#top_clusters <- get_top_clusters(full_data, n = top_cluster_number)
#final_plot <- plot_academic_detailed_map(top_clusters, top_cluster_number)
#ggsave("./Output/Graphs/PostProcessing/Academic_Cluster_Map.pdf", final_plot, width = 8, height = 8, device = cairo_pdf)

#zoom_map <- plot_downtown_zoom_map(top_clusters, top_cluster_number)
#ggsave("./Output/Graphs/PostProcessing/Zoom_Map.pdf", zoom_map, width = 5, height = 6, device = cairo_pdf)

# --- TOP 5 By Severity ---
#top_clusters <- get_top_fatal_infra_clusters(full_data, n = top_cluster_number)
#zoom_map <- plot_high_severity_clusters(top_clusters, top_cluster_number)
#ggsave("./Output/Graphs/PostProcessing/Zoom_Map_Severity.pdf", zoom_map, width = 5, height = 6, device = cairo_pdf)

#final_plot <- plot_infrastructure_priority_map(top_clusters, top_cluster_number)
#ggsave("./Output/Graphs/PostProcessing/Severity_Cluster_Map.pdf", final_plot, width = 8, height = 8, device = cairo_pdf)

message("Execution Flow Completed")