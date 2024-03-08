library(ggplot2)
library(scales)


RLmodel_names <- c("RW", "RW+bias", "RW+bias+Pavlovian(fixed)", "RW+bias+Pavlovian(flexible)")

RLmodel_LOOICs <- c(8201.53, 7960, 7947.84, 7863.79)

RLmodel_table <- data.frame(RLmodel_names, RLmodel_LOOICs)

# bar_colors <- c("#C7582C", "#524F4D", "#387DBD", "#6F17B3")
bar_colors <- c("#524F4D", "#524F4D", "#524F4D", "#524F4D")
# Add the bar_colors column to RLmodel_table
RLmodel_table$bar_colors <- bar_colors

ggplot(data = RLmodel_table, aes(x = RLmodel_names, y = RLmodel_LOOICs, fill=bar_colors)) +
  geom_bar(stat = "identity", width = 0.75) +
  coord_flip() +
  geom_text(aes(label = RLmodel_LOOICs), hjust = 1.2, colour = "white", fontface = "bold") +
  labs(x = "\n Model", y = "LOOIC \n", title = "Model comparison \n") +
  theme(plot.title = element_text(hjust = 0.5, size=15), 
        axis.title.x = element_text(face="bold", colour="black", size = 12),
        axis.title.y = element_text(face="bold", colour="black", size = 12), 
        axis.text = element_text(size = 12))+
  scale_x_discrete(limits = rev(RLmodel_table$RLmodel_names)) +
  scale_y_continuous(limits=c(7750,8250),oob = rescale_none)+
  scale_fill_identity()
 



RLDDMmodel_names <- c("RW", "RW+bias", "RW+bias+Pavlovian(fixed)", "RW+bias+Pavlovian(flexible)")

RLDDMmodel_LOOICs <- c(12539.14, 12303.00, 12296.38, 12205.52)

RLDDMmodel_table <- data.frame(RLmodel_names, RLmodel_LOOICs)

# bar_colors <- c("#C7582C", "#524F4D", "#387DBD", "#6F17B3")
bar_colors <- c("#524F4D", "#524F4D", "#524F4D", "#524F4D")
# Add the bar_colors column to RLDDMmodel_table
RLmodel_table$bar_colors <- bar_colors

ggplot(data = RLDDMmodel_table, aes(x = RLDDMmodel_names, y = RLDDMmodel_LOOICs, fill=bar_colors)) +
  geom_bar(stat = "identity", width = 0.75) +
  coord_flip() +
  geom_text(aes(label = RLDDMmodel_LOOICs), hjust = 1.2, colour = "white", fontface = "bold") +
  labs(x = "\n Model", y = "LOOIC \n", title = "Model comparison \n") +
  theme(plot.title = element_text(hjust = 0.5, size=15), 
        axis.title.x = element_text(face="bold", colour="black", size = 12),
        axis.title.y = element_text(face="bold", colour="black", size = 12), 
        axis.text = element_text(size = 12))+
  scale_x_discrete(limits = rev(RLmodel_table$RLmodel_names)) +
  scale_y_continuous(limits=c(12000,12650),oob = rescale_none)+
  scale_fill_identity()