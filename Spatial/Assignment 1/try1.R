# install.packages(c("sf", "dplyr", "ggplot2", "tmap"))
library(sf)
library(ggplot2)
library(dplyr)

# Load data
argentina_map <- st_read("C:\\Workspace\\Data-Science\\Spatial\\arg.shp")
head(argentina_map)

# Create the map
ggplot(data = argentina_map) +
  geom_sf()

