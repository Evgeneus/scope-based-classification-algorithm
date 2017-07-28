# IN/OUT
# (trust_min=0.75, quiz_papers_n=4, cheaters_prop=0.5)
# random cheaters passed: 31.1301347933%
# smart cheaters passed: 65.1398759233%
# workers passed: 83.0384528428%
# 
# (trust_min=1., quiz_papers_n=4, cheaters_prop=0.5)
# random cheaters passed: 6.23740362776%
# smart cheaters passed: 24.0860810047%
# workers passed: 53.1543248724%
# 
# (trust_min=0.75, quiz_papers_n=6, cheaters_prop=0.5)
# random cheaters passed: 10.9850994038%
# smart cheaters passed: 42.057206336%
# workers passed: 71.0914867957%
# 
# (trust_min=1., quiz_papers_n=6, cheaters_prop=0.5)
# random cheaters passed: 1.56520697777%
# smart cheaters passed: 11.7677029049%
# workers passed: 42.8005109721%

data <- matrix(c(31.13, 65.40, 83.04, 
                 6.24, 24.09, 53.15, 
                 10.10, 42.06, 71.10,
                 1.57, 11.77, 42.80),
               ncol=3,byrow=TRUE)
colnames(data) <- c("Rd_cheaters","Smart_cheaters","Workers")
rownames(data) <- c("trsh: 75%\npapers: 4",
                   "trsh: 100%\npapers: 4",
                   "trsh: 75%\npapers: 6",
                   "trsh: 100%\npapers: 6")

barplot(t(data), main="Passed quiz populatinos vs. parameters settings",
        col=c("gray","darkmagenta", "darkolivegreen1"), beside=TRUE,
        legend.text = c('rand cheaters','smart cheaters','trusted workers'),
        ylab = "Proportion from a population", ylim = c(0, 100), border = "dark blue",
        args.legend = list(x ='topright', bty='n'))
