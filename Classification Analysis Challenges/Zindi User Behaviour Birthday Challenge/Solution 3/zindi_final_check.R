library(lightgbm)
library(tidyverse)
library(data.table)
library(lubridate)
library(caret)
library(foreach)
library(ROCR)
library(keras)
library(reticulate)
library(rstudioapi)
py_discover_config('keras')

path_to_competition_files = "D:/ZINDI"
path_to_my_folder = "D:/ZINDI/zindi_final_check"

train = read.csv(paste0(path_to_competition_files, '/train.csv'))
test = read.csv(paste0(path_to_competition_files, '/test.csv'))
comments = read.csv(paste0(path_to_competition_files, '/Comments.csv'))
comppart = read.csv(paste0(path_to_competition_files, '/CompetitionPartipation.csv'))
comp = read.csv(paste0(path_to_competition_files, '/Competitions.csv'))
discuss = read.csv(paste0(path_to_competition_files, '/Discussions.csv'))
subs = read.csv(paste0(path_to_competition_files, '/Submissions.csv'))
users = read.csv(paste0(path_to_competition_files, '/Users.csv'))

colnames(train)[1] = 'UserID'
colnames(test)[1] = 'UserID'

train$cron_month = 12 * train$year + train$month
test$cron_month = 12 * test$year + test$month

TARGET = train[c('UserID', 'cron_month', 'Target')]

train_test = rbind(train[c('UserID', 'cron_month')], test[c('UserID', 'cron_month')])

discuss$disc_cron_month = 12*discuss$DiscDate.Year + discuss$DiscDate.Month
stat_discuss = discuss %>% group_by(UserID, disc_cron_month) %>% summarize(cnt_disc = length(DiscID),
                                                                           unique_disc = length(unique(DiscID)),
                                                                           sum_F = sum(FeatureF))

comp$CompEndTime.Year = as.numeric(ifelse(comp$CompEndTime.Year == 'not mapped', 10,comp$CompEndTime.Year))
comp$CompEndTime.Month = as.numeric(ifelse(is.na(comp$CompEndTime.Month) == T, 12,comp$CompEndTime.Month ))
comp$comp_cron_month = 12*comp$CompEndTime.Year + comp$CompEndTime.Month

comp$FeatureA_1 = ifelse(comp$FeatureA %like% '1' & !(comp$FeatureA %like% '10'), 1, 0)
comp$FeatureA_2 = ifelse(comp$FeatureA %like% '2', 1, 0)
comp$FeatureA_3 = ifelse(comp$FeatureA %like% '3', 1, 0)
comp$FeatureA_4 = ifelse(comp$FeatureA %like% '4', 1, 0)
comp$FeatureA_5 = ifelse(comp$FeatureA %like% '5', 1, 0)
comp$FeatureA_6 = ifelse(comp$FeatureA %like% '6', 1, 0)

subs = left_join(subs, comp[c('CompID', 'Kind', 'FeatureD', 'comp_cron_month')])
subs$sub_cron_month = 12*subs$SubDate.Year + subs$SubDate.Month
stat_subs = subs %>% group_by(UserID, sub_cron_month) %>% summarize(cnt_subs = length(CompID),
                                                                    unique_comps = length(unique(CompID)),
                                                                    cnt_kind0 = length(CompID[Kind == 0]),
                                                                    cnt_kind1 = length(CompID[Kind == 1]),
                                                                    cnt_d1 = length(CompID[FeatureD == 1]),
                                                                    cnt_d2 = length(CompID[FeatureD == 2]),
                                                                    cnt_d3 = length(CompID[FeatureD == 3]))

comments$comm_cron_month = 12*comments$CommentDate.Year + comments$CommentDate.Month
stat_comments = comments %>% group_by(UserID,  comm_cron_month) %>% summarize(cnt_comments = length(CommentDate.Day_of_week))

comppart = left_join(comppart, comp[c('CompID', 'Kind', 'FeatureD', 'comp_cron_month', 
                                      'FeatureA_1', 'FeatureA_2', 'FeatureA_3', 'FeatureA_4', 'FeatureA_5', 'FeatureA_6')])
comppart$comppart_cron_month = 12*comppart$CompPartCreated.Year + comppart$CompPartCreated.Month
stat_comppart = comppart %>% group_by(UserID,  comppart_cron_month) %>% summarize(cnt_comps_reg = length(unique(CompID)))

start_comps = comp %>% group_by(CompStartTime.Year, CompStartTime.Month) %>% summarize(start_comps = length(unique(CompID))) %>% data.frame()
start_comps$cron_month = 12*start_comps$CompStartTime.Year + start_comps$CompStartTime.Month

start_comps_by_geo = comp %>% group_by(CompStartTime.Year, CompStartTime.Month, Country) %>% summarize(start_comps_geo = length(unique(CompID))) %>% data.frame()
start_comps_by_geo$cron_month = 12*start_comps_by_geo$CompStartTime.Year + start_comps_by_geo$CompStartTime.Month

end_comps = comp %>% group_by(CompEndTime.Year, CompEndTime.Month) %>% summarize(end_comps = length(unique(CompID)))
end_comps$CompEndTime.Year = as.numeric(ifelse(end_comps$CompEndTime.Year == 'not mapped', 10,end_comps$CompEndTime.Year))
end_comps$CompEndTime.Month = as.numeric(ifelse(is.na(end_comps$CompEndTime.Month) == T, 12,end_comps$CompEndTime.Month ))
end_comps$cron_month = 12*end_comps$CompEndTime.Year + end_comps$CompEndTime.Month

users$reg_month = 12*users$UserDate.Year + users$UserDate.Month
users$Country_lab = as.numeric(as.factor(users$Country))
users$Points = as.numeric(as.factor(users$Points))

generate_features <- function(lag, last_n)
{
  trainx_disc_month = left_join(train_test, stat_discuss) %>% filter(is.na(disc_cron_month) == F & 
                                                                       disc_cron_month < cron_month - lag &
                                                                       disc_cron_month >= cron_month - lag - last_n) %>%
    arrange(disc_cron_month) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_disc_month = length(cnt_disc),
      max_disc = max(cnt_disc),
      min_disc = min(cnt_disc),
      first_disc = first(cnt_disc),
      last_disc = last(cnt_disc),
      last_month_disc = last(disc_cron_month)
    ) %>% data.frame()
  
  trainx_disc_all = left_join(train_test, discuss) %>% filter(is.na(disc_cron_month) == F & 
                                                                disc_cron_month < cron_month - lag  &
                                                                disc_cron_month >= cron_month - lag  - last_n) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_disc = length(DiscID),
      sum_F = sum(FeatureF)
    ) %>% data.frame()
  
  trainx_subs_month = left_join(train_test, stat_subs) %>% filter(is.na(sub_cron_month) == F & 
                                                                    sub_cron_month < cron_month - lag  &
                                                                    sub_cron_month >= cron_month - lag  - last_n) %>%
    arrange(sub_cron_month) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_subs_month = length(cnt_subs),
      max_subs = max(cnt_subs),
      min_subs = min(cnt_subs),
      first_subs = first(cnt_subs),
      last_subs = last(cnt_subs),
      last_month_sub = last(sub_cron_month)
    ) %>% data.frame()
  
  trainx_subs_all = left_join(train_test, subs) %>% filter(is.na(sub_cron_month) == F & 
                                                             sub_cron_month < cron_month - lag  &
                                                             sub_cron_month >= cron_month - lag  - last_n) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_subs = length(CompID),
              cnt_comps = length(unique(CompID)),
              cnt_kind0 = length(CompID[Kind == 0]),
              cnt_kind1 = length(CompID[Kind == 1]),
              cnt_d1 = length(CompID[FeatureD == 1]),
              cnt_d2 = length(CompID[FeatureD == 2]),
              cnt_d3 = length(CompID[FeatureD == 3])
    ) %>% data.frame()
  
  trainx_comm_month = left_join(train_test, stat_comments) %>% filter(is.na(comm_cron_month) == F & 
                                                                        comm_cron_month < cron_month - lag  &
                                                                        comm_cron_month >= cron_month - lag  - last_n) %>%
    arrange(comm_cron_month) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_comms_month = length(cnt_comments),
      max_comms = max(cnt_comments),
      min_comms = min(cnt_comments),
      first_comms = first(cnt_comments),
      last_comms = last(cnt_comments),
      last_month_comm = last(comm_cron_month)
    ) %>% data.frame()
  
  trainx_comm_all = left_join(train_test, comments) %>% filter(is.na(comm_cron_month) == F & 
                                                                 comm_cron_month < cron_month - lag  &
                                                                 comm_cron_month >= cron_month - lag  - last_n) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_comms = length(CommentDate.Year)
    ) %>% data.frame()
  
  trainx_comppart_month = left_join(train_test, stat_comppart) %>% filter(is.na(comppart_cron_month) == F & 
                                                                            comppart_cron_month < cron_month - lag  &
                                                                            comppart_cron_month >= cron_month - lag  - last_n) %>%
    arrange(comppart_cron_month) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_comps_reg_month = length(cnt_comps_reg),
      max_comps_reg = max(cnt_comps_reg),
      min_comps_reg = min(cnt_comps_reg),
      first_comps_reg = first(cnt_comps_reg),
      last_comps_reg = last(cnt_comps_reg),
      last_month_comps_reg = last(comppart_cron_month) 
    ) %>% data.frame()
  
  trainx_comppart_all = left_join(train_test, comppart) %>% filter(is.na(comppart_cron_month) == F & 
                                                                     comppart_cron_month < cron_month - lag  &
                                                                     comppart_cron_month >= cron_month - lag  - last_n) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_comps_reg = length(CompID),
      cnt_rank0 = length(CompID[PublicRank == ""]),
      cnt_rank1 = length(CompID[PublicRank == "rank 1"]),
      cnt_rank2 = length(CompID[PublicRank == "rank 2"]),
      cnt_rank3 = length(CompID[PublicRank == "rank 3"]),
      cnt_rank4 = length(CompID[PublicRank == "rank 4"]),
      cnt_rank5 = length(CompID[PublicRank == "rank 5"]),
      cnt_rank6 = length(CompID[PublicRank == "rank 6"]),
      cnt_rank7 = length(CompID[PublicRank == "rank 7"]),
      cnt_rank8 = length(CompID[PublicRank == "rank 8"]),
      cnt_rank9 = length(CompID[PublicRank == "rank 9"]),
      cnt_rank10 = length(CompID[PublicRank == "rank 10"]),
      cnt_rank11 = length(CompID[PublicRank == "rank 11"]),
      cnt_part_kind0 = length(CompID[Kind == 0]),
      cnt_part_kind1 = length(CompID[Kind == 1]),
      cnt_part_d1 = length(CompID[FeatureD == 1]),
      cnt_part_d2 = length(CompID[FeatureD == 2]),
      cnt_part_d3 = length(CompID[FeatureD == 3]),
    ) %>% data.frame()
  
  trainx_all = left_join(train_test[c('UserID', 'cron_month')], trainx_disc_month)
  trainx_all = left_join(trainx_all, trainx_disc_all)
  trainx_all = left_join(trainx_all, trainx_subs_month)
  trainx_all = left_join(trainx_all, trainx_subs_all)
  trainx_all = left_join(trainx_all, trainx_comm_month)
  trainx_all = left_join(trainx_all, trainx_comm_all)
  trainx_all = left_join(trainx_all, trainx_comppart_month)
  trainx_all = left_join(trainx_all, trainx_comppart_all)
  
  trainx_all
  
}

generate_features2 <- function(lag, last_n) {
  
  trainx_subs_all0 = left_join(train_test, subs) %>% filter(is.na(sub_cron_month) == F & 
                                                              sub_cron_month < cron_month - lag  &
                                                              sub_cron_month >= cron_month - lag  - last_n &
                                                              comp_cron_month == cron_month) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_subs_end_month0 = length(CompID)) %>% data.frame()
  trainx_subs_all1 = left_join(train_test, subs) %>% filter(is.na(sub_cron_month) == F & 
                                                              sub_cron_month < cron_month - lag  &
                                                              sub_cron_month >= cron_month - lag  - last_n &
                                                              comp_cron_month == cron_month + 1) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_subs_end_month1 = length(CompID)) %>% data.frame()
  trainx_subs_all2 = left_join(train_test, subs) %>% filter(is.na(sub_cron_month) == F & 
                                                              sub_cron_month < cron_month - lag  &
                                                              sub_cron_month >= cron_month - lag  - last_n &
                                                              comp_cron_month == cron_month + 2) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_subs_end_month2 = length(CompID)) %>% data.frame()
  
  trainx_subs_all3 = left_join(train_test, subs) %>% filter(is.na(sub_cron_month) == F & 
                                                              sub_cron_month < cron_month - lag  &
                                                              sub_cron_month >= cron_month - lag  - last_n &
                                                              comp_cron_month > cron_month + 2 &
                                                              comp_cron_month <= cron_month + 12) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_subs_end_month3 = length(CompID)) %>% data.frame()
  
  trainx_comppart_all0 = left_join(train_test, comppart) %>% filter(is.na(comppart_cron_month) == F & 
                                                                      comppart_cron_month < cron_month - lag  &
                                                                      comppart_cron_month >= cron_month - lag  - last_n &
                                                                      comp_cron_month == cron_month) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_comppart_end_month0 = length(CompID)) %>% data.frame()
  trainx_comppart_all1 = left_join(train_test, comppart) %>% filter(is.na(comppart_cron_month) == F & 
                                                                      comppart_cron_month < cron_month - lag  &
                                                                      comppart_cron_month >= cron_month - lag  - last_n &
                                                                      comp_cron_month == cron_month + 1) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_comppart_end_month1 = length(CompID)) %>% data.frame()
  trainx_comppart_all2 = left_join(train_test, comppart) %>% filter(is.na(comppart_cron_month) == F & 
                                                                      comppart_cron_month < cron_month - lag  &
                                                                      comppart_cron_month >= cron_month - lag  - last_n &
                                                                      comp_cron_month == cron_month + 2) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_comppart_end_month2 = length(CompID)) %>% data.frame()
  
  trainx_comppart_all3 = left_join(train_test, comppart) %>% filter(is.na(comppart_cron_month) == F & 
                                                                      comppart_cron_month < cron_month - lag  &
                                                                      comppart_cron_month >= cron_month - lag  - last_n &
                                                                      comp_cron_month > cron_month + 2 &
                                                                      comp_cron_month <= cron_month + 12) %>% group_by(UserID, cron_month) %>% 
    summarize(cnt_comppart_end_month3 = length(CompID)) %>% data.frame()
  
  trainx_all = left_join(train_test, trainx_subs_all0)
  trainx_all = left_join(trainx_all, trainx_subs_all1)
  trainx_all = left_join(trainx_all, trainx_subs_all2)
  trainx_all = left_join(trainx_all, trainx_subs_all3)
  trainx_all = left_join(trainx_all, trainx_comppart_all0)
  trainx_all = left_join(trainx_all, trainx_comppart_all1)
  trainx_all = left_join(trainx_all, trainx_comppart_all2)
  trainx_all = left_join(trainx_all, trainx_comppart_all3)
  trainx_all
}

generate_features3 <- function(lag, last_n) {
  trainx_comppart_a = left_join(train_test, comppart) %>% filter(is.na(comppart_cron_month) == F & 
                                                                   comppart_cron_month < cron_month - lag  &
                                                                   comppart_cron_month >= cron_month - lag  - last_n) %>% group_by(UserID, cron_month) %>% 
    summarize(
      cnt_part_a1 = length(CompID[FeatureA_1 == 1]),
      cnt_part_a2 = length(CompID[FeatureA_2 == 1]),
      cnt_part_a3 = length(CompID[FeatureA_3 == 1]),
      cnt_part_a4 = length(CompID[FeatureA_4 == 1]),
      cnt_part_a5 = length(CompID[FeatureA_5 == 1]),
      cnt_part_a6 = length(CompID[FeatureA_6 == 1]),
      team_parts = sum(ifelse(PublicRank == '' & Successful.Submission.Count != '', 1, 0))
    ) %>% data.frame()
  trainx_all = left_join(train_test, trainx_comppart_a)
  trainx_all
}

LAG = 0
trainx_1 = generate_features(LAG,1)
trainx_2 = generate_features(LAG,2)
trainx_3 = generate_features(LAG,3)
trainx_6 = generate_features(LAG,6)
trainx_12 = generate_features(LAG,12)
trainx_36 = generate_features(LAG,36)

colnames(trainx_2)[-c(1,2)] = paste0(colnames(trainx_2)[-c(1,2)], '_2m')
colnames(trainx_3)[-c(1,2)] = paste0(colnames(trainx_3)[-c(1,2)], '_3m')
colnames(trainx_6)[-c(1,2)] = paste0(colnames(trainx_6)[-c(1,2)], '_6m')
colnames(trainx_12)[-c(1,2)] = paste0(colnames(trainx_12)[-c(1,2)], '_12m')
colnames(trainx_36)[-c(1,2)] = paste0(colnames(trainx_36)[-c(1,2)], '_alltime')

trainx_1 = left_join(trainx_1, trainx_2)
trainx_1 = left_join(trainx_1, trainx_3)
trainx_1 = left_join(trainx_1, trainx_6)
trainx_1 = left_join(trainx_1, trainx_12)
trainx_1 = left_join(trainx_1, trainx_36)

trainx_1 = left_join(trainx_1, users[c('UserID', 'Country_lab', 'reg_month', 'FeatureX', 'FeatureY', 'Points')])
trainx_1$life = trainx_1$cron_month - trainx_1$reg_month
trainx_1 = left_join(trainx_1, start_comps[c('cron_month', 'start_comps')])
trainx_1 = left_join(trainx_1, end_comps[c('cron_month', 'end_comps')])
trainx_1 = left_join(trainx_1, TARGET)

train1 = trainx_1[1:dim(train)[1],]
test1 = trainx_1[(dim(train)[1] + 1):(dim(train_test)[1]),]

train1 = left_join(train1, users[c('UserID', 'Country')])
train1 = left_join(train1, start_comps_by_geo[c('Country', 'cron_month', 'start_comps_geo')] )

test1 = left_join(test1, users[c('UserID', 'Country')])
test1 = left_join(test1, start_comps_by_geo[c('Country', 'cron_month', 'start_comps_geo')] )


LAG = 1
trainx_1 = generate_features(LAG,1)
trainx_2 = generate_features(LAG,2)
trainx_3 = generate_features(LAG,3)
trainx_6 = generate_features(LAG,6)
trainx_12 = generate_features(LAG,12)
trainx_36 = generate_features(LAG,36)

colnames(trainx_2)[-c(1,2)] = paste0(colnames(trainx_2)[-c(1,2)], '_2m')
colnames(trainx_3)[-c(1,2)] = paste0(colnames(trainx_3)[-c(1,2)], '_3m')
colnames(trainx_6)[-c(1,2)] = paste0(colnames(trainx_6)[-c(1,2)], '_6m')
colnames(trainx_12)[-c(1,2)] = paste0(colnames(trainx_12)[-c(1,2)], '_12m')
colnames(trainx_36)[-c(1,2)] = paste0(colnames(trainx_36)[-c(1,2)], '_alltime')

trainx_1 = left_join(trainx_1, trainx_2)
trainx_1 = left_join(trainx_1, trainx_3)
trainx_1 = left_join(trainx_1, trainx_6)
trainx_1 = left_join(trainx_1, trainx_12)
trainx_1 = left_join(trainx_1, trainx_36)

trainx_1 = left_join(trainx_1, users[c('UserID', 'Country_lab', 'reg_month', 'FeatureX', 'FeatureY', 'Points')])
trainx_1$life = trainx_1$cron_month - trainx_1$reg_month
trainx_1 = left_join(trainx_1, start_comps[c('cron_month', 'start_comps')])
trainx_1 = left_join(trainx_1, end_comps[c('cron_month', 'end_comps')])
trainx_1 = left_join(trainx_1, TARGET)

train2 = trainx_1[1:dim(train)[1],]
test2 = trainx_1[(dim(train)[1] + 1):(dim(train_test)[1]),]

train2 = left_join(train2, users[c('UserID', 'Country')])
train2 = left_join(train2, start_comps_by_geo[c('Country', 'cron_month', 'start_comps_geo')] )

test2 = left_join(test2, users[c('UserID', 'Country')])
test2 = left_join(test2, start_comps_by_geo[c('Country', 'cron_month', 'start_comps_geo')] )

LAG = 2
trainx_1 = generate_features(LAG,1)
trainx_2 = generate_features(LAG,2)
trainx_3 = generate_features(LAG,3)
trainx_6 = generate_features(LAG,6)
trainx_12 = generate_features(LAG,12)
trainx_36 = generate_features(LAG,36)

colnames(trainx_2)[-c(1,2)] = paste0(colnames(trainx_2)[-c(1,2)], '_2m')
colnames(trainx_3)[-c(1,2)] = paste0(colnames(trainx_3)[-c(1,2)], '_3m')
colnames(trainx_6)[-c(1,2)] = paste0(colnames(trainx_6)[-c(1,2)], '_6m')
colnames(trainx_12)[-c(1,2)] = paste0(colnames(trainx_12)[-c(1,2)], '_12m')
colnames(trainx_36)[-c(1,2)] = paste0(colnames(trainx_36)[-c(1,2)], '_alltime')

trainx_1 = left_join(trainx_1, trainx_2)
trainx_1 = left_join(trainx_1, trainx_3)
trainx_1 = left_join(trainx_1, trainx_6)
trainx_1 = left_join(trainx_1, trainx_12)
trainx_1 = left_join(trainx_1, trainx_36)

trainx_1 = left_join(trainx_1, users[c('UserID', 'Country_lab', 'reg_month', 'FeatureX', 'FeatureY', 'Points')])
trainx_1$life = trainx_1$cron_month - trainx_1$reg_month
trainx_1 = left_join(trainx_1, start_comps[c('cron_month', 'start_comps')])
trainx_1 = left_join(trainx_1, end_comps[c('cron_month', 'end_comps')])
trainx_1 = left_join(trainx_1, TARGET)

train3 = trainx_1[1:dim(train)[1],]
test3 = trainx_1[(dim(train)[1] + 1):(dim(train_test)[1]),]

train3 = trainx_1[1:dim(train)[1],]
test3 = trainx_1[(dim(train)[1] + 1):(dim(train_test)[1]),]

train3 = left_join(train3, users[c('UserID', 'Country')])
train3 = left_join(train3, start_comps_by_geo[c('Country', 'cron_month', 'start_comps_geo')] )

test3 = left_join(test3, users[c('UserID', 'Country')])
test3 = left_join(test3, start_comps_by_geo[c('Country', 'cron_month', 'start_comps_geo')] )

train1$magic = train1$Points/train1$life
train2$magic = train2$Points/train2$life
train3$magic = train3$Points/train3$life

test1$magic = test1$Points/test1$life
test2$magic = test2$Points/test2$life
test3$magic = test3$Points/test3$life

train1_extra = generate_features2(0,3)
train2_extra = generate_features2(1,3)
train3_extra = generate_features2(2,3)

train1 = left_join(train1, train1_extra)
train2 = left_join(train2, train2_extra)
train3 = left_join(train3, train3_extra)

test1 = left_join(test1, train1_extra)
test2 = left_join(test2, train2_extra)
test3 = left_join(test3, train3_extra)

train1_extra2 = generate_features3(0,36)
train2_extra2 = generate_features3(1,36)
train3_extra2 = generate_features3(2,36)

train1 = left_join(train1, train1_extra2)
train2 = left_join(train2, train2_extra2)
train3 = left_join(train3, train3_extra2)

test1 = left_join(test1, train1_extra2)
test2 = left_join(test2, train2_extra2)
test3 = left_join(test3, train3_extra2)

geo_stat = users %>% group_by(Country_lab) %>% summarize(Points_country = mean(Points), 
                                                         count_geo = length(UserID), 
                                                         mean_FeatureY_geo = mean(FeatureY))

train1 = left_join(train1, geo_stat)
train2 = left_join(train2, geo_stat)
train3 = left_join(train3, geo_stat)

test1 = left_join(test1, geo_stat)
test2 = left_join(test2, geo_stat)
test3 = left_join(test3, geo_stat)

include = read.csv(paste0(path_to_my_folder, '/sub27_cols.csv'))$x

param_lgb= list(objective = "binary",
                max_bin = 256,
                learning_rate = 0.005,
                num_leaves = 63,
                bagging_fraction = 0.7,
                feature_fraction = 0.7,
                min_data = 100,
                bagging_freq = 1,
                metric = "auc")

# Validation of models, you could skip this part so it was commented

# dtrain <- lgb.Dataset(as.matrix(train1[train1$cron_month >=34 & train1$cron_month <=45,][c(include)]),
#                       label = train1[train1$cron_month >=34 & train1$cron_month <=45,]$Target)
# dtest <- lgb.Dataset(as.matrix(train1[train1$cron_month >=46 & train1$cron_month <=48,][c(include)]),
#                      label = train1[train1$cron_month >=46 & train1$cron_month <=48,]$Target)
# valids = list(train = dtrain, test = dtest)
# model_lgb_imp1 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=10000,
#                            eval_freq = 100, early_stopping_rounds = 500, bagging_seed = 42, feature_fraction_seed = 13) #0.9010 --> #0.9016
# 
# 
# dtrain <- lgb.Dataset(as.matrix(train2[train2$cron_month >=34 & train2$cron_month <=45,][c(include)]),
#                       label = train2[train2$cron_month >=34 & train2$cron_month <=45,]$Target)
# dtest <- lgb.Dataset(as.matrix(train2[train2$cron_month >=46 & train2$cron_month <=48,][c(include)]),
#                      label = train2[train2$cron_month >=46 & train2$cron_month <=48,]$Target)
# valids = list(train = dtrain, test = dtest)
# model_lgb_imp2 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=10000,
#                            eval_freq = 100, early_stopping_rounds = 500, bagging_seed = 13, feature_fraction_seed = 42) #0.9010 --> #0.9016
# 
# 
# dtrain <- lgb.Dataset(as.matrix(train3[train3$cron_month >=34 & train3$cron_month <=45,][c(include)]),
#                       label = train3[train3$cron_month >=34 & train3$cron_month <=45,]$Target)
# dtest <- lgb.Dataset(as.matrix(train3[train3$cron_month >=46 & train3$cron_month <=48,][c(include)]),
#                      label = train3[train3$cron_month >=46 & train3$cron_month <=48,]$Target)
# valids = list(train = dtrain, test = dtest)
# model_lgb_imp3 = lgb.train(data=dtrain, valids = valids, params = param_lgb, nrounds=10000,
#                            eval_freq = 100, early_stopping_rounds = 500, bagging_seed = 13, feature_fraction_seed = 42) #0.9010 --> #0.9016
# 
# pr1 = predict(model_lgb_imp1, as.matrix(train1[train1$cron_month == 46,][c(include)]))
# pr2 = predict(model_lgb_imp2, as.matrix(train2[train2$cron_month == 47,][c(include)]))
# pr3 = predict(model_lgb_imp3, as.matrix(train3[train3$cron_month == 48,][c(include)]))
# 
# pr_cv =  c(pr1,pr2,0.7*pr3)
# fact_cv = c(train1[train1$cron_month == 46,]$Target,train1[train1$cron_month == 47,]$Target, train1[trainx_1$cron_month == 48,]$Target)
# pred_ROCR <- prediction(pr_cv, fact_cv)
# auc.tmp <- performance(pred_ROCR,"auc")
# auc.tmp@y.values

pred1 = 0
dtrain = lgb.Dataset(as.matrix(train1[train1$cron_month >=37 & train1$cron_month <=48,][c(include)]),
                     label = train1[train1$cron_month >=37 & train1$cron_month <=48,]$Target)
for (i in c(1:5)) {
  message(paste0('pred first month: ', i, " iteration of 5"))
  model_lgb_imp = lgb.train(data=dtrain, params = param_lgb, nrounds=1700, bagging_seed = 13+i, feature_fraction_seed = 42+i)
  lgb.save(model_lgb_imp, paste0(path_to_my_folder, '/models/sub27_model1_', i))
  pred_iter = predict(model_lgb_imp, as.matrix(test1[test1$cron_month == 49,][c(colnames(train1[c(include)]))]))
  pred1 = pred_iter + pred1
} 

pred2 = 0
dtrain = lgb.Dataset(as.matrix(train2[train2$cron_month >=37 & train2$cron_month <=48,][c(include)]),
                     label = train2[train2$cron_month >=37 & train2$cron_month <=48,]$Target)
for (i in c(1:5)) {
  message(paste0('pred second month: ', i, " iteration of 5"))
  model_lgb_imp = lgb.train(data=dtrain, params = param_lgb, nrounds=1700, bagging_seed = 42+i, feature_fraction_seed = 13+i)
  lgb.save(model_lgb_imp, paste0(path_to_my_folder, '/models/sub27_model2_', i))
  pred_iter = predict(model_lgb_imp, as.matrix(test2[test2$cron_month == 50,][c(colnames(train2[c(include)]))]))
  pred2 = pred_iter + pred2
} 


pred3 = 0
dtrain = lgb.Dataset(as.matrix(train3[train3$cron_month >=37 & train3$cron_month <=48,][c(include)]),
                     label = train3[train3$cron_month >=37 & train3$cron_month <=48,]$Target)
for (i in c(1:5)) {
  message(paste0('pred third month: ', i, " iteration of 5"))
  model_lgb_imp = lgb.train(data=dtrain, params = param_lgb, nrounds=1700, bagging_seed = 1+i, feature_fraction_seed = 2+i)
  lgb.save(model_lgb_imp, paste0(path_to_my_folder, '/models/sub27_model3_', i))
  pred_iter = predict(model_lgb_imp, as.matrix(test3[test3$cron_month == 51,][c(colnames(train3[c(include)]))]))
  pred3 = pred_iter + pred3
} 

sub27 = data.frame(UserMonthYear = c(paste0(test[test$cron_month == 49,]$UserID, "_", test[test$cron_month == 49,]$month, "_", 
                                                        test[test$cron_month == 49,]$year),
                                                 paste0(test[test$cron_month == 50,]$UserID, "_", test[test$cron_month == 50,]$month, "_", 
                                                        test[test$cron_month == 50,]$year),
                                                 paste0(test[test$cron_month == 51,]$UserID, "_", test[test$cron_month == 51,]$month, "_", 
                                                        test[test$cron_month == 51,]$year)),
                               Target = c(pred1/5, pred2/5, 0.85*pred3/5)) %>% arrange(UserMonthYear)

include1 =  read.csv(paste0(path_to_my_folder, '/perm_imp43_1.csv'))$feature[1:50]
include2 =  read.csv(paste0(path_to_my_folder, '/perm_imp43_2.csv'))$feature[1:50]
include3 =  read.csv(paste0(path_to_my_folder, '/perm_imp43_3.csv'))$feature[1:50]

pred1 = 0
dtrain = lgb.Dataset(as.matrix(train1[train1$cron_month >=37 & train1$cron_month <=48,][c(include1)]),
                     label = train1[train1$cron_month >=37 & train1$cron_month <=48,]$Target)
for (i in c(1:5)) {
  message(paste0('pred first month: ', i, " iteration of 5"))
  model_lgb_imp = lgb.train(data=dtrain, params = param_lgb, nrounds=1700, bagging_seed = 13+i, feature_fraction_seed = 42+i)
  lgb.save(model_lgb_imp, paste0(path_to_my_folder, '/models/sub43_model1_', i))
  pred_iter = predict(model_lgb_imp, as.matrix(test1[test1$cron_month == 49,][c(colnames(train1[c(include1)]))]))
  pred1 = pred_iter + pred1
} 

pred2 = 0
dtrain = lgb.Dataset(as.matrix(train2[train2$cron_month >=37 & train2$cron_month <=48,][c(include2)]),
                     label = train2[train2$cron_month >=37 & train2$cron_month <=48,]$Target)
for (i in c(1:5)) {
  message(paste0('pred second month: ', i, " iteration of 5"))
  model_lgb_imp = lgb.train(data=dtrain, params = param_lgb, nrounds=1700, bagging_seed = 42+i, feature_fraction_seed = 13+i)
  lgb.save(model_lgb_imp, paste0(path_to_my_folder, '/models/sub43_model2_', i))
  pred_iter = predict(model_lgb_imp, as.matrix(test2[test2$cron_month == 50,][c(colnames(train2[c(include2)]))]))
  pred2 = pred_iter + pred2
} 


pred3 = 0
dtrain = lgb.Dataset(as.matrix(train3[train3$cron_month >=37 & train3$cron_month <=48,][c(include3)]),
                     label = train3[train3$cron_month >=37 & train3$cron_month <=48,]$Target)
for (i in c(1:5)) {
  message(paste0('pred third month: ', i, " iteration of 5"))
  model_lgb_imp = lgb.train(data=dtrain, params = param_lgb, nrounds=1700, bagging_seed = 1+i, feature_fraction_seed = 2+i)
  lgb.save(model_lgb_imp, paste0(path_to_my_folder, '/models/sub43_model3_', i))
  pred_iter = predict(model_lgb_imp, as.matrix(test3[test3$cron_month == 51,][c(colnames(train3[c(include3)]))]))
  pred3 = pred_iter + pred3
} 


sub43 = data.frame(UserMonthYear = c(paste0(test[test$cron_month == 49,]$UserID, "_", test[test$cron_month == 49,]$month, "_", 
                                            test[test$cron_month == 49,]$year),
                                     paste0(test[test$cron_month == 50,]$UserID, "_", test[test$cron_month == 50,]$month, "_", 
                                            test[test$cron_month == 50,]$year),
                                     paste0(test[test$cron_month == 51,]$UserID, "_", test[test$cron_month == 51,]$month, "_", 
                                            test[test$cron_month == 51,]$year)),
                   Target = c(pred1/5, pred2/5, 0.85*pred3/5)) %>% arrange(UserMonthYear)

include_nn = read.csv(paste0(path_to_my_folder, '/mlp_cols.csv'), header = T)$Feature

train0 = train1[c(include_nn)]
for (i in c(1:240))
  train0[,i] = ifelse(is.na(train0[,i]) == TRUE, 0, train0[,i])
for (i in c(1:240))
  train0[,i] = ifelse(is.infinite(train0[,i]) == TRUE, 0, train0[,i])
maxs1_1 <- apply(train0[-c(1,219,220)],2,max)
mins1_1 <- apply(train0[-c(1,219,220)],2,min)
scaled1 <- data.frame(target = train0$Target, cron_month = train0$cron_month,
                      scale(train0[-c(1,219,220)],center = mins1_1, scale = maxs1_1 - mins1_1))

train0 = train2[c(include_nn)]
for (i in c(1:240))
  train0[,i] = ifelse(is.na(train0[,i]) == TRUE, 0, train0[,i])
for (i in c(1:240))
  train0[,i] = ifelse(is.infinite(train0[,i]) == TRUE, 0, train0[,i])
maxs1_2 <- apply(train0[-c(1,219,220)],2,max)
mins1_2 <- apply(train0[-c(1,219,220)],2,min)
scaled2 <- data.frame(target = train0$Target, cron_month = train0$cron_month,
                      scale(train0[-c(1,219,220)],center = mins1_2, scale = maxs1_2 - mins1_2))

train0 = train3[c(include_nn)]
for (i in c(1:240))
  train0[,i] = ifelse(is.na(train0[,i]) == TRUE, 0, train0[,i])
for (i in c(1:240))
  train0[,i] = ifelse(is.infinite(train0[,i]) == TRUE, 0, train0[,i])
maxs1_3 <- apply(train0[-c(1,219,220)],2,max)
mins1_3 <- apply(train0[-c(1,219,220)],2,min)
scaled3 <- data.frame(target = train0$Target, cron_month = train0$cron_month,
                      scale(train0[-c(1,219,220)],center = mins1_3, scale = maxs1_3 - mins1_3))

#you can run this to reproduce NN-models after uncommenting, but you can immediately use ready weights in section below comments
# for (i in c(1:10)) {
#   model_checking <- callback_model_checkpoint(paste0(path_to_my_folder, '/keras/model1_final/', i+30,'w.{epoch:02d}-{val_loss:.6f}.hdf5'),
#                                               monitor = "val_loss", verbose = 0, save_best_only = TRUE, save_weights_only = FALSE, mode = c("auto"), save_freq = "epoch")
#   set.seed(i)
#   model <- keras_model_sequential()
#   model %>%
#     layer_dense(units = 80, activation = 'relu', input_shape = c(237)) %>%
#     layer_dense(units = 40, activation = 'relu') %>%
#     #layer_dense(units = 10, activation = 'elu') %>%
#     layer_dense(units = 1, activation = 'sigmoid')
#   model %>% compile(
#     loss = 'binary_crossentropy',
#     optimizer = 'adam',
#     metrics = 'binary_crossentropy'
#   )
#   model %>% fit(x = as.matrix(scaled1[scaled1$cron_month <=47,][-c(1,2)]), y = as.matrix(scaled1[scaled1$cron_month <=47,]$target),
#                 validation_data = list(as.matrix(scaled1[scaled1$cron_month == 48,][-c(1,2)]), as.matrix(scaled1[scaled1$cron_month == 48,]$target)),
#                 epochs = 30, batch_size = 128, verbose = 0, callbacks = c(model_checking))
# }
# 
# for (i in c(1:10)) {
#   model_checking <- callback_model_checkpoint(paste0(path_to_my_folder, '/keras/model2_final/',i+30,'w.{epoch:02d}-{val_loss:.6f}.hdf5'),
#                                               monitor = "val_loss", verbose = 0, save_best_only = TRUE, save_weights_only = FALSE, mode = c("auto"), save_freq = "epoch")
#   set.seed(i)
#   model <- keras_model_sequential()
#   model %>%
#     layer_dense(units = 80, activation = 'relu', input_shape = c(237)) %>%
#     layer_dense(units = 40, activation = 'relu') %>%
#     #layer_dense(units = 10, activation = 'elu') %>%
#     layer_dense(units = 1, activation = 'sigmoid')
#   model %>% compile(
#     loss = 'binary_crossentropy',
#     optimizer = 'adam',
#     metrics = 'binary_crossentropy'
#   )
#   model %>% fit(x = as.matrix(scaled2[scaled2$cron_month <=46,][-c(1,2)]), y = as.matrix(scaled2[scaled2$cron_month <=46,]$target),
#                 validation_data = list(as.matrix(scaled2[scaled2$cron_month == 48,][-c(1,2)]), as.matrix(scaled2[scaled2$cron_month == 48,]$target)),
#                 epochs = 30, batch_size = 128, verbose = 0, callbacks = c(model_checking))
# }
# 
# for (i in c(1:10)) {
#   model_checking <- callback_model_checkpoint(paste0(path_to_my_folder, '/keras/model3_final/',i+30,'w.{epoch:02d}-{val_loss:.6f}.hdf5'),
#                                               monitor = "val_loss", verbose = 0, save_best_only = TRUE, save_weights_only = FALSE, mode = c("auto"), save_freq = "epoch")
#   set.seed(i)
#   model <- keras_model_sequential()
#   model %>%
#     layer_dense(units = 80, activation = 'relu', input_shape = c(237)) %>%
#     layer_dense(units = 40, activation = 'relu') %>%
#     #layer_dense(units = 10, activation = 'elu') %>%
#     layer_dense(units = 1, activation = 'sigmoid')
#   model %>% compile(
#     loss = 'binary_crossentropy',
#     optimizer = 'adam',
#     metrics = 'binary_crossentropy'
#   )
#   model %>% fit(x = as.matrix(scaled3[scaled3$cron_month <=45,][-c(1,2)]), y = as.matrix(scaled3[scaled3$cron_month <=45,]$target),
#                 validation_data = list(as.matrix(scaled3[scaled3$cron_month == 48,][-c(1,2)]), as.matrix(scaled3[scaled3$cron_month == 48,]$target)),
#                 epochs = 30, batch_size = 128, verbose = 0, callbacks = c(model_checking))
# }

test1 = test1[c(include_nn)]
test2 = test1[c(include_nn)]
test3 = test1[c(include_nn)]

for (i in c(1:240)) {
  test1[,i] = ifelse(is.na(test1[,i]) == TRUE, 0, test1[,i])
  test1[,i] = ifelse(is.infinite(test1[,i]) == TRUE, 0, test1[,i])
  test2[,i] = ifelse(is.na(test2[,i]) == TRUE, 0, test2[,i])
  test2[,i] = ifelse(is.infinite(test2[,i]) == TRUE, 0, test2[,i])
  test3[,i] = ifelse(is.na(test3[,i]) == TRUE, 0, test3[,i])
  test3[,i] = ifelse(is.infinite(test3[,i]) == TRUE, 0, test3[,i])
}

scaled_test1 <- data.frame(cron_month = test1$cron_month, scale(test1[-c(1,219,220)],center = mins1_1, scale = maxs1_1 - mins1_1))
scaled_test2 <- data.frame(cron_month = test2$cron_month, scale(test2[-c(1,219,220)],center = mins1_2, scale = maxs1_2 - mins1_2))
scaled_test3 <- data.frame(cron_month = test3$cron_month, scale(test3[-c(1,219,220)],center = mins1_3, scale = maxs1_3 - mins1_3))

### These are already trained NN-models with the weights (if you train them by yourself in above commented section of code, you should change filenames according to received validation scores of BEST 3 models got on training section)
nn1_1 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model1_final/17w.12-0.146775.hdf5'), compile = FALSE)
nn1_2 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model1_final/20w.06-0.146994.hdf5'), compile = FALSE) 
nn1_3 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model1_final/1w.11-0.147074.hdf5'), compile = FALSE) 

nn2_1 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model2_final/13w.07-0.159511.hdf5'), compile = FALSE)
nn2_2 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model2_final/29w.10-0.159869.hdf5'), compile = FALSE) 
nn2_3 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model2_final/9w.11-0.160161.hdf5'), compile = FALSE) 

nn3_1 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model3_final/28w.06-0.166977.hdf5'), compile = FALSE)
nn3_2 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model3_final/22w.09-0.166999.hdf5'), compile = FALSE) 
nn3_3 = load_model_hdf5(paste0(path_to_my_folder, '/keras/model3_final/7w.15-0.167069.hdf5'), compile = FALSE) 

pr_nn1_1 = predict(nn1_1, as.matrix(scaled_test1[scaled_test1$cron_month == 49,][-c(1)]))
pr_nn1_2 = predict(nn1_2, as.matrix(scaled_test1[scaled_test1$cron_month == 49,][-c(1)]))
pr_nn1_3 = predict(nn1_3, as.matrix(scaled_test1[scaled_test1$cron_month == 49,][-c(1)]))

pr_nn2_1 = predict(nn2_1, as.matrix(scaled_test2[scaled_test2$cron_month == 50,][-c(1)]))
pr_nn2_2 = predict(nn2_2, as.matrix(scaled_test2[scaled_test2$cron_month == 50,][-c(1)]))
pr_nn2_3 = predict(nn2_3, as.matrix(scaled_test2[scaled_test2$cron_month == 50,][-c(1)]))

pr_nn3_1 = predict(nn3_1, as.matrix(scaled_test3[scaled_test3$cron_month == 51,][-c(1)]))
pr_nn3_2 = predict(nn3_2, as.matrix(scaled_test3[scaled_test3$cron_month == 51,][-c(1)]))
pr_nn3_3 = predict(nn3_3, as.matrix(scaled_test3[scaled_test3$cron_month == 51,][-c(1)]))

pr_nn1_final = (pr_nn1_1 + pr_nn1_2 + pr_nn1_3)/3
pr_nn2_final = (pr_nn2_1 + pr_nn2_2 + pr_nn2_3)/3
pr_nn3_final = (pr_nn3_1 + pr_nn3_2 + pr_nn3_3)/3


sub_nn1 = data.frame(UserMonthYear = c(paste0(test[test$cron_month == 49,]$UserID, "_", test[test$cron_month == 49,]$month, "_", 
                                              test[test$cron_month == 49,]$year),
                                       paste0(test[test$cron_month == 50,]$UserID, "_", test[test$cron_month == 50,]$month, "_", 
                                              test[test$cron_month == 50,]$year),
                                       paste0(test[test$cron_month == 51,]$UserID, "_", test[test$cron_month == 51,]$month, "_", 
                                              test[test$cron_month == 51,]$year)),
                     Target = c(pr_nn1_final, pr_nn2_final, pr_nn3_final)) %>% arrange(UserMonthYear)

sub54 = data.frame(UserMonthYear = sub27$UserMonthYear, Target = 0.375*rank(sub43$Target) + 0.375*rank(sub27$Target) + 0.25*rank(sub_nn1$Target))
write.csv(sub54, paste0(path_to_my_folder, '/zindi_final_check_sub54_new.csv'), row.names = F, quote = F)