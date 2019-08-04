# ライブラリ読み込み
library(cvTools)
library(Metrics)
library(ROCR)
library(MLmetrics)
library(glmnet)
library(Matching)
library(ranger)
library(edarf)
library(data.table)

# 読み込み / 設定
df <-read.csv("https://raw.githubusercontent.com/iwanami-datascience/vol3/master/kato%26hoshino/q_data_x.csv")
df_cp <- df
print(dim(df))
not_use_cols <- c("gamecount", "gamedummy", "gamesecond")
df_model <- df[, !(colnames(df) %in% not_use_cols)]
print(dim(df_model))

# 5CVで傾向スコアの学習
k <- 5
folds <- cvFolds(NROW(df), K = k)
df$oof_pred <- rep(0, nrow(df))
for (i in 1:k) {
  print(paste0("CV", i))
  train <- df_model[folds$subsets[folds$which != i], ]
  validation <- df_model[folds$subsets[folds$which == i], ]
  # Note: 本来はglmnetなどを用いて、正則化した方が良い
  # Note: 書籍では1行後のように、共変量を絞っている
  model <- glm(formula = as.factor(cm_dummy) ~ ., data = train, family=binomial)
  model <-glm(cm_dummy ~ TVwatch_day + age + sex + marry_dummy + child_dummy + inc + pmoney + area_kanto +area_tokai + area_keihanshin + job_dummy1 + job_dummy2 + job_dummy3 + job_dummy4 + job_dummy5 + job_dummy6 + job_dummy7  + fam_str_dummy1 + fam_str_dummy2 + fam_str_dummy3 + fam_str_dummy4, family=binomial(link="logit") , data = df_model)
  valid_pred <- predict(model, validation, type="response")
  df[folds$subsets[folds$which == i], ]$oof_pred <- valid_pred
}

# loglossとaucの確認
print(LogLoss(df$oof, df$cm_dummy))
oof_train_auc <- prediction(df$oof, df$cm_dummy)
tmp.auc <- performance(oof_train_auc, "auc")
print(as.numeric(tmp.auc@y.values))

png_path = "./ps_score.png"
png(png_path)
par(mfrow=c(2,1))
hist(df[df$cm_dummy==1,]$oof_pred, main="test", xlab="PS", xlim=c(0,1), col=rgb(0,0,1,0.5),breaks=seq(0,1,0.01))
hist(df[df$cm_dummy==0,]$oof_pred, main="ctrl", xlab="PS", xlim=c(0,1), col=rgb(1,0,0,0.5), breaks=seq(0,1,0.01))
dev.off()

# 傾向スコアを用いたマッチング
caliper <- 0.2
M <- 1
ties <- FALSE
match <- Match(Y=df$gamesecond, 
               Tr=df$cm_dummy,
               X=df$oof_pred,
               estimand = "ATT",
               M=M,
               caliper = caliper,
               ties=ties)
print(summary(match))

# マッチングしたデータのペアのデータフレームの作成
df_tmp<- df
df_tmp$id <- 1:nrow(df_tmp)
df_pair <- cbind(df_tmp[match$index.treated, c('id', colnames(df))],
                 df_tmp[match$index.control, c('id',colnames(df))])
test_cols <- list()
ctr_cols <- list()
for (i in 1 :length(colnames(df))){
  new_name <- paste0("test_", colnames(df)[i])
  test_cols[i] <- new_name
  new_name <- paste0("ctr_", colnames(df)[i])
  ctr_cols[i] <- new_name
}
colnames(df_pair) = c('test_id', test_cols, 'ctr_id', ctr_cols)


# SDの計算
for (col in c("TVwatch_day", "inc", "pmoney")){
  test_colname <- paste0("test_", col)
  ctr_colname <- paste0("ctr_", col)
  mean_test <- mean(df_pair[, test_colname])
  mean_ctr <- mean(df_pair[, ctr_colname])
  var_test <- var(df_pair[, test_colname])
  var_ctr <- var(df_pair[, ctr_colname])
  SD <- abs((mean_test - mean_ctr) /  sqrt((var_test + var_ctr) / 2))
  print(paste0("SD_of_", col, "_is:", SD))
}


# IPWの計算
# IPWの計算 ATTの式になっているので注意
print("caluculating_IPW...")
Y <- df$gamesecond
Z <- df$cm_dummy
PS <- df$oof_pred

IPW1 <-sum(Z * Y / PS) / sum(Z / PS)
IPW0 <- sum(((1 - Z) * Y) / (1 - PS)) / sum((1 - Z) / (1 -PS))
effect <- IPW1 - IPW0
print(paste(IPW0, IPW1, effect, sep="_"))
print(paste0("effect", effect))

# 回帰モデル
x_col <- colnames(df)
x_col <- x_col[-which(x_col %in% c("gamesecond", "gamecount", "cm_dummy","gamedummy", "oof_pred"))]
# z=0, 1でデータセットを分割
df_0 <- df[df$cm_dummy==0, ]
df_1 <- df[df$cm_dummy==1, ]

# 学習
model_0 <- cv.glmnet(y=df_0$gamesecond, x=as.matrix(df_0[x_col]), family ="gaussian", nfolds=5,  alpha=0)
model_1 <- cv.glmnet(y=df_1$gamesecond, x=as.matrix(df_1[x_col]), family ="gaussian", nfolds=5,  alpha=0)

# 0, 1への予測とmseの算出
pred_00 <- predict(model_0, s="lambda.min", newx=as.matrix(df_0[x_col]))
print(mean((df_0$gamesecond - pred_00)^2))
pred_11 <-  predict(model_1, s="lambda.min", newx=as.matrix(df_1[x_col]))
print(mean((df_1$gamesecond - pred_11)^2))

# ダブルロバスト
Y <- df$gamesecond
Z <- df$cm_dummy
PS <- df$oof_pred
reg_0 <- predict(model_0, s="lambda.min", newx=as.matrix(df[x_col]))
reg_1 <- predict(model_1, s="lambda.min", newx=as.matrix(df[x_col]))

# E(y_1)
dr_1 <- mean(Z * Y / PS + (1 - Z / PS) * reg_1)
# E(y_0)
dr_0 <- mean((1 - Z) * Y / (1 - PS) + (1 - (1 - Z) / (1 - PS)) * reg_0)

print(paste0("E(y_1):", dr_1, "_E(y_0):", dr_0, "_effect:", dr_1-dr_0))


# Proximityマッチング
# モデリング -> Proximityマトリックスの算出
model <- ranger(formula = as.factor(cm_dummy) ~ ., data = df_cp, num.trees = 500)
proximity_matrix <- extract_proximity(model, newdata=df_cp)
print(dim(proximity_matrix))

# # 対角成分を削除して、testのみの行 / ctrのみの列にする
prox_matrix_remove_diag <- proximity_matrix - diag(nrow(proximity_matrix))
test_index <- df$cm_dummy == 1
prox_matrix_remove_diag_test <- prox_matrix_remove_diag[test_index, !test_index]
print(dim(prox_matrix_remove_diag_test))

# #  各テスト群のデータ(行)に対して最も類似度の高いコントロール群(列)の抽出
max_index <- apply(prox_matrix_remove_diag_test, 1, which.max)
df_pair_test <- df[test_index, ]
df_pair_ctr <- df[max_index, ]
# 因果効果
effect <- mean(df_pair_test$gamesecond) - mean(df_pair_ctr$gamesecond)
print(paste0("effect_is_", effect))

# SDの計算
for (col in c("TVwatch_day", "inc", "pmoney")){
  mean_test <- mean(df_pair_test[, col])
  mean_ctr <- mean(df_pair_ctr[, col])
  var_test <- var(df_pair_test[, col])
  var_ctr <- var(df_pair_ctr[, col])
  SD <- abs((mean_test - mean_ctr) /  sqrt((var_test + var_ctr) / 2))
  print(paste0("SD_of_", col, "_is:", SD))
}
