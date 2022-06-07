



############ Grouping Functions
group_age <- function(age){
  if (age >= 0 & age <= 25){
    return('students')
  }else if(age > 25 & age <= 40){
    return('strong working class')
  }else if (age > 40 & age <= 55){
    return('weak working class')
  }else if (age > 55){
    return('grand parents')
  }
}

group_size <- function(x){
  if (x > 0 & x <= 3){
    return('small family')
  }else if(x > 3 & x <= 6){
    return('average family')
  }else if (x > 6 & x <= 9){
    return('large family')
  }else{
    return('exlarge family')
  }
}

group_edu <- function(x){
  if (x > 0 & x <= 4){
    return('bin1')
  }else if(x > 4 & x <= 8){
    return('bin2')
  }else if (x > 8 & x <= 12){
    return('bin3')
  }else{
    return('bin4')
  }
}

group_ex = function(x){
  if(is.na(x)){
   return("None")
  }else if(x >= 0 & x <= 20){
    return('bin2')
  }else if (x > 20 & x <= 44){
    return('bin3')
  }else{
    return('bin4')
  }
}


# Encode a character or factor with its hexavegisimal value
hexv.encode = function(x, xnew = x) {
  lvls = as.character(unique(x))
  lvls = lvls[order(nchar(lvls), tolower(lvls))]
  return (as.integer(factor(xnew, levels = lvls)))
}

# Encode a feature value with its frequency in the entire dataset
freq.encode = function(x, xnew = x) {
  if (is.factor(x) || is.character(x)) {
    return (as.numeric(factor(xnew, levels = names(sort(table(x))))))
  } else {
    return (approxfun(density(x[!is.na(x)], n = length(x) / 100))(xnew))
  }
}


####
SimpleEnsemble = function(pred1,pred2,weight = 0.5){
  pred = weight*pred1 + (1 -weight) *pred2
  return(pred)
}
#########

