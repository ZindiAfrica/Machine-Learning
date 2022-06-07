#######################
### LOADING DATA
#######################
df = read.csv(paste0(path.dir,"/train.csv"))
test = read.csv(paste0(path.dir,"/test.csv"))
label = df$depressed

test.id = test$surveyid


###################
### Look at the properties of the features
###################
ftrs = data.frame(
  type      = unlist(lapply(df, class)),
  n.unique  = unlist(lapply(df, function(x) length(unique(x)))),
  f.missing = unlist(lapply(df, function(x) mean(is.na(x)))),
  spear.cor  = unlist(lapply(df, function(x) { idx = !is.na(x); 
  if (is.factor(x)) x = as.numeric(x);
  if (is.integer(x)) x = as.numeric(x);
  if (is.character(x)) x = as.numeric(x);
  cor(x[idx], y = as.numeric(label)[idx], method = 'spearman') }))
)
##### Dropping no variation features
ftrs$name = rownames(ftrs)
ftrs = ftrs %>% drop_na()
df = df[,names(df) %in% ftrs$name]

### MISSING VALUES
#check for columns with missing values
na.cols <- which(colSums(is.na(df)) > 0)
na.cols <- sort(colSums(sapply(df[na.cols], is.na)), decreasing = TRUE)
paste('There are', length(na.cols), 'columns with missing values')


####################
### DATA CLEANING AND PROCESSING 
####################

##### Age
df$age = round(df$age)
test$age[test$age == ".d"] = 17
test$age[test$age == ""] = 17
test$age = round(as.numeric(test$age))
### grouping age
df$age2 <- sapply(df$age, group_age)
df$age2 <- as.factor(df$age2) %>% as.numeric()
test$age2 = sapply(test$age, group_age)
test$age2 = as.factor(test$age2) 
test$age2 = as.numeric(test$age2)

##### age freq Encode
df$age_encode = freq.encode(df$age)
test$age_encode = freq.encode(test$age)

### hh size
df$hhsize2 = sapply(df$hhsize, group_size)
df$hhsize2 = as.factor(df$hhsize2) %>% as.numeric()
test$hhsize2 = sapply(test$hhsize, group_size)
test$hhsize2 = as.factor(test$hhsize2) %>% as.numeric()

#### grouping Education
df$edu2 = sapply(df$edu, group_edu)
df$edu2 = as.factor(df$edu2) %>% as.numeric()
test$edu2 = sapply(test$edu, group_edu)
test$edu2 = as.factor(test$edu2) %>% as.numeric()

#### fsadwholed_often
df$fs_adwholed_often = round(df$fs_adwholed_often)
test$fs_adwholed_often = round(test$fs_adwholed_often)
### FE of 0 values in fsadwholed_often
df$fs_has0 = ifelse(df$fs_adwholed_often == 0,1,0)
test$fs_has0 = ifelse(test$fs_adwholed_often == 0,1,0)

##### med_u4_death
df$med_u5_deaths = round(df$med_u5_deaths)
test$med_u5_deaths = round(test$med_u5_deaths)
df$med_hasNA = ifelse(is.na(df$med_u5_deaths),1,0)
test$med_hasNA = ifelse(is.na(test$med_u5_deaths),1,0)
### school attend
df$ed_schoolattend = round(df$ed_schoolattend)
test$ed_schoolattend = round(test$ed_schoolattend)
df$ed_schoolattend = ifelse(df$ed_schoolattend >=1, 1,0)
df$has_Na = ifelse(is.na(df$ed_schoolattend),1,0)
test$has_Na = ifelse(is.na(test$ed_schoolattend),1,0)

### Cleaning survey date feature
### some of which are irrelevant
df$survey_date= as.character(df$survey_date)
date.format <- as.Date(as.character(df$survey_date), format="%d-%b-%Y")
df$survey_date = as.Date(df$survey_date, format="%d-%b-%Y")
df$survey_date = df$survey_date %>% gsub(pattern = "00", replacement = "19")
df$year = year(df$survey_date) %>% as.factor() %>% as.numeric()
df$month = as.factor(months(date.format)) %>% as.numeric()
df$weekdays = as.factor(weekdays(date.format)) %>% as.numeric()
df$quarters = quarters(date.format) %>% as.factor() %>% as.numeric()
df$Date_week <- as.integer(strftime(df$survey_date, format="%W"))
#####
test$survey_date= as.character(test$survey_date)
date.format <- as.Date(test$survey_date, format="%d-%b-%Y")
test$survey_date = as.Date(test$survey_date, format="%d-%b-%Y")
test$survey_date = test$survey_date %>% gsub(pattern = "00", replacement = "19")
test$year = year(test$survey_date) %>% as.factor() %>% as.numeric()
test$month = as.factor(months(date.format)) %>% as.numeric()
test$weekdays = as.factor(weekdays(date.format)) %>% as.numeric()
test$quarters = quarters(date.format) %>% as.factor() %>% as.numeric()
test$Date_week <- as.integer(strftime(test$survey_date, format="%W"))




########################
### EXTRA DATA CLEANING AND FEATURE ENGINEERING
### No enough time to experiment, uncomment to try
########################
# ##### ed expenses
# df$ed_expenses = round(df$ed_expenses)
# test$ed_expenses = round(test$ed_expenses)
# df$ed_expenses[df$ed_expenses>100] = 100
# test$ed_expenses[test$ed_expenses>100] = 100

# df$ed_expenses2 = sapply(df$ed_expenses,group_ex)
# df$ed_expenses2 = as.factor(df$ed_expenses2) %>% as.numeric()
# test$ed_expenses2 = sapply(test$ed_expenses, group_ex)
# test$ed_expenses2 = as.factor(test$ed_expenses) %>% as.numeric()

# ### durable investment
# df$durable_investment = round(df$durable_investment)
# df$durable_investment[df$durable_investment > 400] = 400
# df$durable_investment[df$durable_investment > 0 & df$durable_investment <=100] = 100
# df$durable_investment[df$durable_investment > 100 & df$durable_investment <=300] = 200
# df$durable_investment[df$durable_investment > 200 & df$durable_investment <=399] = 300
# test$durable_investment = round(test$durable_investment)
# test$durable_investment[test$durable_investment > 400] = 400
# test$durable_investment[test$durable_investment > 0 & test$durable_investment <=100] = 100
# test$durable_investment[test$durable_investment > 100 & test$durable_investment <=300] = 200
# test$durable_investment[test$durable_investment > 200 & test$durable_investment <=399] = 300

# ## ed work act
# df$ed_work_act_pc = round(df$ed_work_act_pc)
# df$ed_work_act_pc = ifelse(df$ed_work_act_pc >0,1,0)
# test$ed_work_act_pc = round(test$ed_work_act_pc)
# test$ed_work_act_pc = ifelse(test$ed_work_act_pc>0,1,0)
# 
# ###m med health consilt
# df$med_healthconsult = round(df$med_healthconsult)
# test$med_healthconsult = round(test$med_healthconsult)
# #### med sick days
# df$med_sickdays_hhave = round(df$med_sickdays_hhave)
# test$med_sickdays_hhave = round(test$med_sickdays_hhave)
# 
# ### cons social
# df$cons_social = round(df$cons_social)
# df$cons_social[df$cons_social>4] = 4
# test$cons_social = round(test$cons_social)
# test$cons_social[test$cons_social>4] = 4
# 
# df$cons_social_is0 = ifelse(df$cons_social == 0,1,0)
# test$cons_social_is0 = ifelse(test$cons_social == 0,1,0)
# 
# ### asset savings
# df$asset_savings = round(df$asset_savings)
# df$asset_savings[df$asset_savings > 16] = 16
# test$asset_savings = round(test$asset_savings)
# test$asset_savings[test$asset_savings>16] = 16
# 
# df$asset_savings_has0 = ifelse(df$asset_savings == 0,1,0)
# test$asset_savings_has0 = ifelse(test$asset_savings == 0,1,0)
# 
# ### asset livestock
# df$asset_livestock = round(df$asset_livestock)
# df$asset_livestock[df$asset_livestock>384] = 384
# test$asset_livestock = round(test$asset_livestock)
# test$asset_livestock[test$asset_livestock>384] = 384
# 
# df$lhas0 = ifelse(df$asset_livestock == 0,1,0)
# test$lhas0 = ifelse(test$asset_livestock == 0,1,0)
# 
# # asset phone
# df$asset_phone = round(df$asset_phone)
# df$asset_phone = ifelse(df$asset_phone>0,1,0)
# test$asset_phone = round(test$asset_phone)
# test$asset_phone = ifelse(test$asset_phone>0,1,0)
# 
# ## cons nondurable
# df$cons_nondurable =round(df$cons_nondurable)
# group_con <- function(x){
#   if (x > 0 & x <= 200){
#     return('bin1')
#   }else if(x > 200 & x <= 400){
#     return('bin2')
#   }else{
#     return('bin4')
#   }
# }
# df$con = sapply(df$cons_nondurable,group_con)
# df$con = as.factor(df$con) %>% as.numeric()
# test$con = sapply(test$cons_nondurable, group_con)
# test$con = as.factor(test$con) %>% as.numeric()
# 


### DROPPING IRRELEVANT FEATURES TO START WITH
df = df[,-c(1:4)]
df = df %>% select(-depressed)
features.names = names(df)
test = test[, names(test) %in% features.names]

## replacing NA values with -1
df[is.na(df)] = -1
test[is.na(test)] = -1


### setting labels for probablities prediction
label2 = ifelse(label == 0, "No","Yes")

