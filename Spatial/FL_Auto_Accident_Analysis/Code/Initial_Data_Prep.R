library(data.table)

# Using fread for speed on 7.7M rows
df <- fread("./Datasets/US_Accidents_March23.csv")

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

fwrite(fl_combined, "./Datasets/Florida_Combined_Data.csv")

cat("Final count of Florida records:", nrow(fl_combined), "\n")