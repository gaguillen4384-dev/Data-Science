library(data.table)
library(sf)
library(dplyr)
library(leaflet)
library(htmlwidgets)


# --- SUBSET FUNCTIONS ---

get_florida_subset <- function(original_dataset, output_florida_dataset){
# Using fread for speed on 7.7M rows
df <- fread(original_dataset)

# Define Florida's geographic boundaries (Bounding Box)
# Min/Max Lat: ~24.4 to 31.0
# Min/Max Lng: ~-87.6 to -80.0
lat_min <- 24.396308
lat_max <- 31.000889
lng_min <- -87.634896
lng_max <- -80.031362

fl_combined <- df[
  State == "FL" | 
    (Start_Lat >= lat_min & Start_Lat <= lat_max & 
       Start_Lng >= lng_min & Start_Lng <= lng_max)
]

fwrite(fl_combined, output_florida_dataset)

cat("Final count of Florida records:", nrow(fl_combined), "\n")
}

get_central_florida_subset <- function(florida_dataset, output_cfl_dataset){
  # Load the full Florida dataset
  df <- fread(florida_dataset)
  
  # Define Orange County specific boundaries (Approximate)
  lat_min <- 28.33 
  lat_max <- 28.78 
  lng_min <- -81.75 
  lng_max <- -80.95 
  
  # Filter strictly for Orange County
  # use & (AND) for coordinates to ensure they fall WITHIN the box
  orange_county_df <- df[
    (County == "Orange") | 
      (Start_Lat >= lat_min & Start_Lat <= lat_max & 
         Start_Lng >= lng_min & Start_Lng <= lng_max)
  ]
  
  # regardless of what the lat/long says:
  # orange_county_df <- orange_county_df[County == "Orange"]
  
  # These are the records that will have to map via 'County/Street' logic later
  missing_lat_long_subset <- orange_county_df[is.na(Start_Lat) | is.na(Start_Lng), .N]
  
  # Save the specific subset
  fwrite(orange_county_df, output_cfl_dataset)
  
  # --- Logging Output ---
  cat("--------------------------------------------\n")
  cat("Subset Summary for Orange County\n")
  cat("--------------------------------------------\n")
  cat("Total records in subset:          ", nrow(orange_county_df), "\n")
  cat("Records missing Lat/Long:         ", missing_lat_long_subset, "\n")
  cat("Percentage missing Lat/Long:      ", 
      round((missing_lat_long_subset / max(1, nrow(orange_county_df))) * 100, 2), "%\n")
  cat("--------------------------------------------\n")
}

get_central_florida_shape_subset <- function(fdot_state_road, output_cfl_state_road){
  roads_sf <- st_read(fdot_state_road)
  
  # use %in% to select multiple counties at once
  roads_central_fl <- roads_sf %>%
    filter(COUNTY %in% c("Orange"))
  
  st_write(roads_central_fl, output_cfl_state_road, delete_dsn = TRUE)
  cat("Success! shapefile created.\n")
}

# --- ENRICHMENT FUNCTIONS ---

enrich_accident_data <- function(accident_path, shapefile_path, output_path) {
  cat("Loading accident data...\n")
  accidents_raw <- fread(accident_path)
  
  # Remove records with missing lat/long (they can't be joined spatially)
  accidents_spatial <- accidents_raw[!is.na(Start_Lat) & !is.na(Start_Lng)]
  cat("Records ready for spatial join:", nrow(accidents_spatial), "\n")
  
  # Convert Accidents to a Spatial Object (SF)
  # Using CRS 4326 (standard GPS coordinates)
  accidents_sf <- st_as_sf(accidents_spatial, 
                           coords = c("Start_Lng", "Start_Lat"), 
                           crs = 4326,
                           remove = FALSE)
  
  cat("Loading and preparing shapefile...\n")
  roads_sf <- st_read(shapefile_path)
  
  # Ensure both datasets use the exact same Coordinate Reference System
  # transform accidents to match the road file's projection
  accidents_sf <- st_transform(accidents_sf, st_crs(roads_sf))
  
  # Perform the Spatial Join (Nearest Neighbor)
  # st_join with st_nearest_feature finds the closest road for every point
  cat("Performing spatial join (this may take a minute)...\n")
  enriched_sf <- st_join(accidents_sf, roads_sf, join = st_nearest_feature)
  
  # convert back to a regular dataframe/datatable to save as CSV
  cat("Saving enriched data to:", output_path, "\n")
  enriched_df <- as.data.table(enriched_sf)
  enriched_df[, geometry := NULL]
  
  fwrite(enriched_df, output_path)
  
  cat("Success! Enriched file created with", ncol(enriched_df), "columns.\n")
}

view_cfl_map <- function(enriched_path, output_filename = "./Output/Graphs/orange_county_map.html"){
  enriched_df <- fread(enriched_path)
  
  # Ensure Severity is treated as a factor or numeric for the palette
  # Change 'SEVERITY' to the exact column name in your CSV
  enriched_sf <- st_as_sf(enriched_df, 
                          coords = c("Start_Lng", "Start_Lat"), 
                          crs = 4326)
  
  # Define a color palette: 1 (Low) to 4 (High)
  # Using a colorblind-friendly/soft palette
  sev_pal <- colorFactor(
    palette = c("#440154", "#31688e", "#35b779", "#fde725"), # Viridis: Purple to Yellow
    domain = enriched_sf$SEVERITY
  )
  
  # Create the map object
  m <- leaflet(enriched_sf) %>%
    addProviderTiles(providers$CartoDB.Positron) %>% # Using a lighter, cleaner base map
    addCircleMarkers(
      radius = 4,
      fillOpacity = 0.8,
      weight = 1,
      color = "white", # White border makes markers pop
      fillColor = ~sev_pal(Severity), 
      popup = ~paste("<b>Severity:</b>", Severity, "<br>",
                     "<b>Road:</b>", RouteNum, "<br>",
                     "<b>County:</b>", County, "<br>",
                     "<b>Milepost:</b>", BEGIN_POST, "-", END_POST)
    ) %>%
    addLegend(
      pal = sev_pal, 
      values = ~Severity, 
      title = "Severity Level",
      position = "bottomright"
    )
  
  # Save the map
  saveWidget(m, file = output_filename, selfcontained = TRUE)
  
  message(paste("Map successfully saved to", output_filename))
}

# --- EXECUTION FLOW ---
orginal_dataset <- "./Datasets/Original/US_Accidents_March23.csv"
#"./Datasets/test_test.csv"
florida_dataset <- "./Datasets/Original/Florida_Combined_Data.csv "
cfl_dataset <- "./Datasets/Central_Florida_Combined_Data.csv"
fdot_state_road <- "./Datasets/Original/State_Roads.shp"
cfl_state_road <- "./Datasets/Central_FL_Roads.shp"
enriched_cfl_dataset <- "./Datasets/Enriched_Central_Florida_Combined_Data.csv"

# Do this to get 7.7 million to only Florida-data accidents
#get_florida_subset(original_dataset=orginal_dataset,
#                   output_florida_dataset = florida_dataset)

# Do this to split up the FDOT state road data into central florida
#get_central_florida_shape_subset(fdot_state_road= fdot_state_road,
#                                 output_cfl_state_road = cfl_state_road)

# Do this to get the CFL accident dataset
#get_central_florida_subset(florida_dataset = florida_dataset,
#                           output_cfl_dataset = cfl_dataset)

# Do this to add state road info to CFL accident dataset
#enrich_accident_data(accident_path = cfl_dataset,
#                     shapefile_path = cfl_state_road,
#                     output_path =enriched_cfl_dataset)

# Do this to visualize, it might explode CPU usage
#view_cfl_map(enriched_cfl_dataset)