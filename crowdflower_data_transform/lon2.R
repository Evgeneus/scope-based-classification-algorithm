criteria_parser <- function(criteria) {
  elderly_radio <- 'yes';
  si_radio <- 'yes';
  tech_radio <- 'yes';
  
  if (grepl('not OA', criteria) | grepl('no OA', criteria)) {
    elderly_radio <- 'no';
  }
  if (grepl('not SI', criteria) | grepl('no SI', criteria)) {
    si_radio <- 'no';
  }
  if (grepl('not tech', criteria) | grepl('no tech', criteria)){
    tech_radio <- 'no';
  }
  
  gold <- c(elderly_radio, si_radio, tech_radio);
  return(gold)
}


full <- lon2_995784
sources <- source995784

worker_ids <- unique(full$X_worker_id);
paper_ids <- source995784$paper_id;


#create a dataframe
empty_data <- matrix(nrow = length(worker_ids), ncol = length(paper_ids));
data <- cbind(worker_ids, empty_data);
row_names <- c('worker_id', paper_ids);
colnames(data) <- row_names;
data <- as.data.frame(data);

#filling out the 'data'
criteria <- c('elderly_radio', 'si_radio', 'tech_radio');

for (worker_id in worker_ids){
  w_data <- full[full$X_worker_id==worker_id,];
  w_papers <- w_data$paper_title;
  for (paper_title in w_papers){
    paper_id_temp <- sources[sources$paper_title==paper_title,]
    if (nrow(paper_id_temp) <= 0) {return('Error paper_id_temp')}
    paper_id <- paper_id_temp$paper_id;
    if (paper_id <= 0) {return('eeeror')}
    
    votes <- w_data[w_data$paper_title==paper_title,];
    if (nrow(votes) <= 0) {return('Error votes')}
    w_elderly_radio <- toString(votes[['elderly_radio']]);
    w_si_radio <- toString(votes[['si_radio']]);
    w_tech_radio <- toString(votes[['tech_radio']]);
    
    w_count <- 0;
    criteria_gold <- criteria_parser(toString(votes$criteria));
    if (w_elderly_radio == criteria_gold[1]) {w_count <- w_count + 1}
    if (w_si_radio == criteria_gold[2]) {w_count <- w_count + 1}
    if (w_tech_radio == criteria_gold[3]) {w_count <- w_count + 1}

    data[data$worker_id==worker_id,][paper_id+1] <- w_count;
  }
}

write.csv(data, 'lon2_agg.csv')
print('Done')