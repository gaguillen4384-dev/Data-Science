#Author: gaguillen4384-dev

library(sf)
library(leaflet)
library(dplyr)


spatial_data <- st_read("geo_export_36807f63-b0c3-4978-b6ac-5ea15c157a66.shp")

nyc_points <- spatial_data %>%
  st_transform(2263) %>%
  st_centroid() %>%
  st_transform(4326) %>%
  mutate(
    popup_text = paste0(
      "<strong>Zip Code (MODZCTA): </strong>", modzcta, "<br>",
      "<strong>Estimated Population: </strong>", format(pop_est, big.mark=","), "<br>",
      "<strong>Included ZCTAs: </strong>", label
    ) %>% lapply(htmltools::HTML),
    
    population_category = ifelse(pop_est >= median(pop_est, na.rm = TRUE), 
                          "High Population Area", "Lower Population Area")
  )


population_pallete <- colorNumeric(
  palette = "RdYlBu", 
  domain = nyc_points$pop_est,
  na.color = "transparent"
)


leaflet(nyc_points) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  
  addCircleMarkers(
    data = filter(nyc_points, population_category == "High Population Area"),
    group = "High Population Areas",
    radius = ~sqrt(pop_est)/20,      
    color = ~population_pallete(pop_est),   
    stroke = TRUE, weight = 1,
    fillOpacity = 0.8,
    label = ~popup_text
  ) %>%
  
  addCircleMarkers(
    data = filter(nyc_points, population_category == "Lower Population Area"),
    group = "Lower Population Areas",
    radius = ~sqrt(pop_est)/20,
    color = ~population_pallete(pop_est),
    stroke = TRUE, weight = 1,
    fillOpacity = 0.8,
    label = ~popup_text
  ) %>%
  
  addLayersControl(
    overlayGroups = c("High Population Areas", "Lower Population Areas"),
    options = layersControlOptions(collapsed = FALSE)
  ) %>%
  
  addLegend(
    pal = population_pallete, 
    values = ~pop_est, 
    title = "Est. Population",
    position = "bottomright"
  )
