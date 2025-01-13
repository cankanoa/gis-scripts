
## Load required packages 
require(raster)
require(rgdal)
require(ggplot2)
require(tools)
require(effectsize)



## Set main working directory 
main.dir <- "C:/Users/borg/Desktop/WorldView_seagrass/"



## Define input variables 
# Name of field data input folder
field.input <- "Back_Sound_2013"
# Name of satellite data input folder
satellite.input <- "Back_Sound_20130327"
# Average patchy and continuous ranges 
patchy.avg <- 37.5
cont.avg <- 85



## Read in field and satellite data 
back.field.list <- list.files(paste0(main.dir,"R_library/", field.input), "*.shp$", full.names = T)
back.field <- readOGR(dirname(back.field.list), file_path_sans_ext(basename(back.field.list)))
back.sat <- raster(list.files(paste0(main.dir,"Input_data/2_Classification_data/3_Classified_image/", satellite.input), "*.TIF$", recursive = T, full.names = T))



## Create an empty dataframe to populate with results 
back.df <- as.data.frame(matrix(nrow = length(back.field), ncol = 6))
colnames(back.df) <- c("Shapefile ID","Field data","n No data","n No seagrass","n Seagrass","n Valid")
## Populate ID column
back.df$`Shapefile ID` <- back.field$FID_1
## Populate field data column 
back.df$`Field data` <- ifelse(back.field$Class == "PATCHY", patchy.avg, cont.avg)



## Extract satellite classification results for each polygon 
back.sat.polygon <- extract(back.sat, back.field, method = "simple")



## Collapse classes into generalized classes based on their classification values 
# 1 = Seagrass, 2 = Shallow Water, 3 = Deep Water, 4 = Turbid Water, 5 = CDOM water, 6 = Submerged sand, 7 = Land
# No data indicates pixels quality flagged as CDOM or turbid water 
back.df$`n No data` <- unlist(lapply(back.sat.polygon, function(x){length(which(x %in% c(4,5)))}))
# No seagrass indicates water pixels where the substrate was visible but seagrass was not present
back.df$`n No seagrass` <- unlist(lapply(back.sat.polygon, function(x){length(which(x %in% c(2,3,6)))}))
back.df$`n Seagrass` <- unlist(lapply(back.sat.polygon, function(x){length(which(x == 1))}))
# Valid indicates water pixels exlcluding those categorized as no data
back.df$`n Valid` <- back.df$`n No seagrass` + back.df$`n Seagrass`



## Compute satellite-indicated percentage seagrass for each polygon 
back.df$`Satellite data` <- (back.df$`n Seagrass`/back.df$`n Valid` * 100)
## Set polygons with less than 10% of their data as valid to NA
back.df$`Satellite data`[which(((back.df$`n Valid`)/(back.df$`n No data` + back.df$`n Valid`)) < 0.1)] <- NA
## Set polygons with less than 10 pixels to NA
back.df$`Satellite data`[which(back.df$`n Valid` < 10)] <- NA
## Discard any rows that include NAs 
back.df <- back.df[complete.cases(back.df[,c("Field data", "Satellite data")]),]



## Create boxplots of results 
ggplot(back.df, aes(as.factor(`Field data`), `Satellite data`)) + geom_boxplot(fill = "gray90", width = 0.5) +
  theme_classic() + ylab("Satellite-indicated percentage cover") + 
  xlab("Reference-indicated percentage cover") + ggtitle("(c) Back Sound, NC") +
  scale_x_discrete(labels = c(paste0("Patchy (*n* = ", length(which(back.df$Field.data == 37.5)), ")"),
                              paste0("Continuous (*n* = ", length(which(back.df$Field.data == 85)),")"))) + 
  theme(axis.title = element_text(size = 12), axis.text.x = ggtext::element_markdown(size = 10),
        axis.text.y = element_text(size = 12), plot.title = element_text(size = 16))
ggsave(paste0(main.dir,"Assess_Agreement/Polygon_PercentCover/Boxplots/Back_Sound_Boxplots.jpeg"), height = 4, width = 3.5, units = "in", dpi = 300)



## Compute Mann-Whitney U Test
back.mwu <- wilcox.test(subset(back.df, `Field data` == patchy.avg)$`Satellite data`, subset(back.df, `Field data` == cont.avg)$`Satellite data`, alternative = "less")
back.rbs <- rank_biserial(subset(back.df, `Field data` == patchy.avg)$`Satellite data`, subset(back.df, `Field data` == cont.avg)$`Satellite data`, alternative = "less")
## Create output csv file 
back.output <- as.data.frame(matrix(nrow = 2, ncol = 5))
colnames(back.output) <- c("Class","n","Median","Rank-serial","Difference")
back.output$Class <- c("Patchy","Continuous")
back.output$n <- c(length(which(back.df$`Field data` == patchy.avg)),length(which(back.df$`Field data` == cont.avg)))
back.output$Median <- c(median(back.df$`Satellite data`[which(back.df$`Field data` == patchy.avg)], na.rm = T),median(back.df$`Satellite data`[which(back.df$`Field data` == 85)], na.rm = T))
back.output$`Rank-serial` <- back.rbs$r_rank_biserial
back.output$Difference <- ifelse(abs(back.rbs$r_rank_biserial) < 0.1, "Negligible", ifelse(abs(back.rbs$r_rank_biserial) < 0.3, "Small", ifelse(abs(back.rbs$r_rank_biserial) < 0.5, "Moderate", "Large")))

