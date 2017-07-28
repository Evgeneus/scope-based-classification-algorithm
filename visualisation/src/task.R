# QUIZ

# set1 <- as.numeric(accuracy[2,][10:1009])
# set2 <- as.numeric(accuracy[3,][10:1009])
# set3 <- as.numeric(accuracy[4,][10:1009])
# set4 <- as.numeric(accuracy[5,][10:1009])
# set5 <- as.numeric(accuracy[6,][10:1009])
# set6 <- as.numeric(accuracy[7,][10:1009])
# acc <- read.csv('acc_passed_dist.csv');
acc <- acc_passed_z03_t3_7
set1 <- na.omit(as.numeric(acc[1,])) # test_items=4 z=0.1
set2 <- na.omit(as.numeric(acc[2,])) # test_items=4 z=0.25
set3 <- na.omit(as.numeric(acc[3,])) # test_items=4 z=0.5
set4 <- na.omit(as.numeric(acc[4,])) # test_items=4 z=0.75
set5 <- na.omit(as.numeric(acc[5,])) # test_items=4 z=0.9

# par(mfrow=c(2,3))
# hist(set1, col="grey", main = paste("accuracies dist\nz=0.1, tests=8"), prob=TRUE)
# lines(density(set1), col="blue", lwd=1.5)
# hist(set2, col="grey", main = paste("accuracies dist\n z=0.25, tests=8"), prob=TRUE)
# lines(density(set2), col="blue", lwd=1.5)
# hist(set3, col="grey", main = paste("accuracies dist\n z=0.5, tests=8"), prob=TRUE)
# lines(density(set3), col="blue", lwd=1.5)
# hist(set4, col="grey", main = paste("accuracies dist\n z=0.75, tests=8"), prob=TRUE)
# lines(density(set4), col="blue", lwd=1.5)
# hist(set5, col="grey", main = paste("accuracies dist\nz=0.9, tests=8"), prob=TRUE)
# lines(density(set5), col="blue", lwd=1.5)

# plots
plot(density(set1, from = 0.5, to = 1.), col="blue", lwd=1.5, xlab='Accuracy', 
     main = "Accuracy distribution for tests=4",
     ylim = c(0, 7.5))
lines(density(set2, from = 0.5, to = 1.), col="green", lwd=1.5)
lines(density(set3, from = 0.5, to = 1.), col="brown", lwd=1.5)
lines(density(set4, from = 0.5, to = 1.), col="darkorchid", lwd=1.5)
lines(density(set5, from = 0.5, to = 1.), col="red", lwd=1.5)
# Add a legend
legend(
  "topleft", 
  lty=c(1,1,1,1),
  bty='n',
  col=c("blue", "green", "brown", "darkorchid", "red"), 
  legend = c("z = 0.10", "z = 0.25", "z = 0.50", "z = 0.75", "z = 0.90")
)

# library(ggplot2)
# 
# #Sample data
# dat <- data.frame(dens = c(rnorm(100), rnorm(100, 10, 5))
#                   , lines = rep(c("a", "b"), each = 100))
# #Plot.
# ggplot(dat, aes(x = dens, fill = lines)) + geom_density(alpha = 0.5)




# z=0.3 tests=[3,4,5,6,7]
plot(density(set1), col="blue", lwd=1.5, xlab='Accuracy', 
     main = "Accuracy distribution for z = 0.30",
     ylim = c(0, 7.5))
lines(density(set2), col="green", lwd=1.5)
lines(density(set3), col="brown", lwd=1.5)
lines(density(set4), col="darkorchid", lwd=1.5)
lines(density(set5), col="red", lwd=1.5)
legend(
  "topleft", 
  lty=c(1,1,1,1),
  bty='n',
  col=c("blue", "green", "brown", "darkorchid", "red"), 
  legend = c("tests = 3", "tests = 4", "tests = 5", "tests = 6", "tests = 7")
)



# TASK EXECUTION
# accuracy of results
library(ggplot2)
data <- task_results_plot
l1 <- data[data$judgment_min == 3,]
l2 <- data[data$judgment_min == 5,]
l3 <- data[data$judgment_min == 7,]

l1$judgment = '3'
l2$judgment = '5'
l3$judgment = '7'

acc_plot <- rbind(l1, l2, l3)
pd <- position_dodge(0.1) # move them .05 to the left and right
ggplot(acc_plot, aes(x=papers_page, y=acc_avg, group=judgment, col=judgment, fill=judgment)) + 
  geom_errorbar(aes(ymin=acc_avg-acc_std, ymax=acc_avg+acc_std), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7,8,9)) +
  ggtitle("Accuracy of results VS papers per page") +
  xlab('Papers per page') +
  ylab('Accuracy')


# fp_lose
ggplot(acc_plot, aes(x=papers_page, y=fp_lose_avg, group=judgment, col=judgment, fill=judgment)) + 
  geom_errorbar(aes(ymin=fp_lose_avg-fp_lose_std, ymax=fp_lose_avg+fp_lose_std), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7,8,9)) +
  ggtitle("False positive lose VS papers per page") +
  xlab('Papers per page') +
  ylab('FP lose')

# fn_lose
ggplot(acc_plot, aes(x=papers_page, y=fn_lose_avg, group=judgment, col=judgment, fill=judgment)) + 
  geom_errorbar(aes(ymin=fn_lose_avg-fn_lose_std, ymax=fn_lose_avg+fn_lose_std), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7,8,9)) +
  ggtitle("False negative lose VS papers per page") +
  xlab('Papers per page') +
  ylab('FN lose')

# total_lose
ggplot(acc_plot, aes(x=papers_page, y=(fn_lose_avg+fp_lose_avg), group=judgment, col=judgment, fill=judgment)) + 
  geom_errorbar(aes(ymin=(fn_lose_avg+fp_lose_avg)-(fp_lose_std+fn_lose_std)/2, ymax=(fn_lose_avg+fp_lose_avg)+(fp_lose_std+fn_lose_std)/2), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7,8,9)) +
  ggtitle("Total lose VS papers per page") +
  xlab('Papers per page') +
  ylab('Total lose')

#budget spent
ggplot(acc_plot, aes(x=papers_page, y=budget_spent_avg, group=judgment, col=judgment, fill=judgment)) + 
  # geom_errorbar(aes(ymin=budget_spent_std, ymax=budget_spent_avg+acc_std), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7,8,9)) +
  ggtitle("Budget spent VS papers per page") +
  xlab('Papers per page') +
  ylab('Budget spent, $')


# CLASSIFICATION FUNCTION
library(ggplot2)
data <- task_results_plot
l1 <- data[data$judgment_min == 3,]
l2 <- data[data$judgment_min == 5,]
l3 <- data[data$judgment_min == 7,]

l1$judgment = '3'
l2$judgment = '5'
l3$judgment = '7'

acc_plot <- rbind(l1, l2, l3)
pd <- position_dodge(0.1) # move them .05 to the left and right

data <- task_results_cost
ggplot(acc_plot, aes(x=papers_page, y=(fn_mv_lose_avg+fp_mv_lose_avg), group=judgment, col=judgment, fill=judgment)) + 
  # geom_errorbar(aes(ymin=(fn_lose_avg+fp_lose_avg)-(fp_lose_std+fn_lose_std)/2, ymax=(fn_lose_avg+fp_lose_avg)+(fp_lose_std+fn_lose_std)/2), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7,8,9)) +
  ggtitle("Total lose VS papers per page") +
  xlab('Papers per page') +
  ylab('Total lose')

ggplot(acc_plot, aes(x=papers_page, y=(fn_lose_avg+fp_lose_avg), group=judgment, col=judgment, fill=judgment)) + 
  # geom_errorbar(aes(ymin=(fn_lose_avg+fp_lose_avg)-(fp_lose_std+fn_lose_std)/2, ymax=(fn_lose_avg+fp_lose_avg)+(fp_lose_std+fn_lose_std)/2), width=.1, position=pd) +
  geom_line(position=pd) +
  geom_point(position=pd) +
  scale_x_continuous(breaks=c(1,2,3,4,5,6,7,8,9)) +
  ggtitle("Total lose VS papers per page") +
  xlab('Papers per page') +
  ylab('Total lose')