full <- read.csv('full_report.csv');

worker_ids <- unique(full$worker_id);
paper_ids <- unique(full$paper_id);

#getting test and row ids
test_paper_ids <- c(1165492146,
                    1165492147,
                    1165492150,
                    1167479032,
                    1167481004,
                    1167481536,
                    1167483698,
                    1173377614);
row_ids <- paper_ids[!(paper_ids %in% test_paper_ids)];

#lists consisting from "paper_is_creiteria", criteria: eld, si, tech
criteria_ids <- c();
for (test_id in c(test_paper_ids, row_ids)){
  id_str <- toString(test_id);
  criteria_ids <- c(criteria_ids, c(paste(id_str,'eld', sep='_'), 
                                    paste(id_str,'si', sep='_'), paste(id_str,'tech', sep='_')));
}

#create a dataframe
empty_data <- matrix(nrow = length(worker_ids), ncol = length(paper_ids)*3);
data <- cbind(worker_ids, empty_data);
row_names <- c('worker_id', criteria_ids);
colnames(data) <- row_names;
data <- as.data.frame(data);

#filling out the 'data'
# 1 if a worker was correct
# 0 if a worker was wrang
# NA if a worker did not tag a paper
signs <- c('eld', 'si', 'tech');
criteria <- c('elderly_radio', 'si_radio', 'tech_radio');
names(signs) <- criteria;

for (worker_id in worker_ids){
  w_data <- full[full$worker_id==worker_id,];
  w_papers <- w_data$paper_id;
  for (paper_id in w_papers){
    votes <- w_data[w_data$paper_id==paper_id,];
    for (cf_name in criteria){
      criteria_sign <- signs[[cf_name]];
      worker_tag <- toString(votes[[cf_name]]);
      gold_tag <- toString(votes[[paste(cf_name, 'gold', sep = '_')]]);
      result <- as.integer(grepl(worker_tag, gold_tag));
      column_name <- paste(toString(paper_id), criteria_sign, sep = '_');
      data[data$worker_id==worker_id,][column_name] <- result;
    }
  }
}