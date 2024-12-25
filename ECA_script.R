#installing and loading required packages
install.packages("topicmodels")
install.packages("sentimentr")
install.packages("tidytext")
install.packages("dplyr")
install.packages("ldatuning")
install.packages("LDAvis")
install.packages("servr")

library(tm)
library(SnowballC)
library(topicmodels)
library(slam)
library(sentimentr)
library(tidytext)
library(dplyr)
library(ldatuning)
library(LDAvis)
library(servr)

#loading text file and creating corpus for data frame
txtpath<-file.path("C:/Users/peizh/OneDrive/Desktop/SUSS/Y3S1 Aug'24/ANL312 Text Mining and Applied Project Formulation/ECA/650records_fashion_review.csv")

txt<-read.csv(txtpath)

comments_corpus<-VCorpus(DataframeSource(txt))
comments<-comments_corpus

#data preprocessing

#converting all texts to lower case
comments<-tm_map(comments,content_transformer(tolower))
#removing punctuations from text
comments<-tm_map(comments,removePunctuation)
#removing numbers from text
comments<-tm_map(comments,removeNumbers)
#removing non-English words
toSpace<-content_transformer(function(x,pattern)gsub(pattern," ",x))
comments<-tm_map(comments, toSpace, "[^A-Za-z']")

#remove stopwords based on existing list provided in tm package
comments<-tm_map(comments, removeWords, stopwords("en"))
#converting texts to their base form
comments<-tm_map(comments, stemDocument)
#remove additional spaces
comments<-tm_map(comments, stripWhitespace)

#find out additional frequent occurring words to add to stop list
comments_df <- data.frame(text = sapply(comments, as.character), stringsAsFactors = FALSE)
comments_df %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE)

#second round of removing additional stopwords based on above finding plus relevant fashion terms
mystoplist<-scan(file="C:/Users/peizh/OneDrive/Desktop/SUSS/Y3S1 Aug'24/ANL312 Text Mining and Applied Project Formulation/ECA/custom_stopwords.data",
                 what=character(),sep="\n")
comments<-tm_map(comments,removeWords,mystoplist)

#generate Document-Term Matrix
dtm<-DocumentTermMatrix(comments)
dtm$dimnames$Terms

#checking Document-Term Matrix
dim(dtm)
#create Document-Term Matrix with TF-IDF values
dtm_tfidf<-weightTfIdf(dtm)

inspect(dtm)
summary(dtm_tfidf$v)

#StudyGuide SU5 Chapter 2 Page 125
dtm_reduced<-dtm[,dtm_tfidf$v>=0.22]
dtm_reduced<-dtm_reduced[row_sums(dtm_reduced)>0,]
inspect(dtm_reduced)

#Using metrics to determine k
result <- FindTopicsNumber(
  dtm,
  topics = seq(2, 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 1234),
  mc.cores = 1L,
  verbose = TRUE
)
FindTopicsNumber_plot(result)

#TOPIC MODELLING
#k=6
k<-6
seed<-2020
model_lda_6<-list(VEM=LDA(dtm_reduced,k=k,method="VEM",control=list(seed=seed)),
                Gibbs=LDA(dtm_reduced,k=k,method="Gibbs",control=list(
                  seed=seed,burnin=1000,thin=100,iter=1000)))
#results
terms_vem<-terms(model_lda_6[["VEM"]],10)
terms_gibbs<-terms(model_lda_6[["Gibbs"]],10)
terms_vem
terms_gibbs

#k=7
k<-7
seed<-2020
model_lda_7<-list(VEM=LDA(dtm_reduced,k=k,method="VEM",control=list(seed=seed)),
Gibbs=LDA(dtm_reduced,k=k,method="Gibbs",control=list(seed=seed,burnin=1000,thin=100,iter=1000)))

#results
terms_vem<-terms(model_lda_7[["VEM"]],10)
terms_gibbs<-terms(model_lda_7[["Gibbs"]],10)
terms_vem
terms_gibbs

#k=8
k<-8
seed<-2020
model_lda_8<-list(VEM=LDA(dtm_reduced,k=k,method="VEM",control=list(seed=seed)),
                Gibbs=LDA(dtm_reduced,k=k,method="Gibbs",control=list(
                  seed=seed,burnin=1000,thin=100,iter=1000)))
#results
terms_vem<-terms(model_lda_8[["VEM"]],10)
terms_gibbs<-terms(model_lda_8[["Gibbs"]],10)
terms_vem
terms_gibbs

#k=9
k<-9
seed<-2020
model_lda_9<-list(VEM=LDA(dtm_reduced,k=k,method="VEM",control=list(seed=seed)),
                Gibbs=LDA(dtm_reduced,k=k,method="Gibbs",control=list(
                  seed=seed,burnin=1000,thin=100,iter=1000)))
#results
terms_vem<-terms(model_lda_9[["VEM"]],10)
terms_gibbs<-terms(model_lda_9[["Gibbs"]],10)
terms_vem
terms_gibbs

#k=10
k<-10
seed<-2020
model_lda_10<-list(VEM=LDA(dtm_reduced,k=k,method="VEM",control=list(seed=seed)),
                Gibbs=LDA(dtm_reduced,k=k,method="Gibbs",control=list(
                  seed=seed,burnin=1000,thin=100,iter=1000)))
#results
terms_vem<-terms(model_lda_10[["VEM"]],10)
terms_gibbs<-terms(model_lda_10[["Gibbs"]],10)
terms_vem
terms_gibbs

#first row is DocID, second row onwards is TopicID, to see the topics each Doc contains
topic_vem<-topics(model_lda[["VEM"]],3)
topic_vem[,1:5]

# Extract necessary data for visualization
phi <- posterior(model_lda_6[["Gibbs"]])$terms  # Term-topic probabilities for Gibbs
theta <- posterior(model_lda_6[["Gibbs"]])$topics  # Document-topic proportions for Gibbs
vocab <- colnames(dtm_reduced)  # Vocabulary terms
doc_length <- rowSums(as.matrix(dtm_reduced))  # Document lengths
term_frequency <- colSums(as.matrix(dtm_reduced))  # Term frequencies

json_data <- createJSON(
  phi = phi,
  theta = theta,
  vocab = vocab,
  doc.length = doc_length,
  term.frequency = term_frequency
)

serVis(json_data, out.dir = 'LDAvis', open.browser = TRUE)

# Extract necessary data for visualization
phi <- posterior(model_lda_7[["Gibbs"]])$terms  # Term-topic probabilities for Gibbs
theta <- posterior(model_lda_7[["Gibbs"]])$topics  # Document-topic proportions for Gibbs
vocab <- colnames(dtm_reduced)  # Vocabulary terms
doc_length <- rowSums(as.matrix(dtm_reduced))  # Document lengths
term_frequency <- colSums(as.matrix(dtm_reduced))  # Term frequencies

json_data <- createJSON(
  phi = phi,
  theta = theta,
  vocab = vocab,
  doc.length = doc_length,
  term.frequency = term_frequency
)

serVis(json_data, out.dir = 'LDAvis', open.browser = TRUE)

# Extract necessary data for visualization
phi <- posterior(model_lda_8[["Gibbs"]])$terms  # Term-topic probabilities for Gibbs
theta <- posterior(model_lda_8[["Gibbs"]])$topics  # Document-topic proportions for Gibbs
vocab <- colnames(dtm_reduced)  # Vocabulary terms
doc_length <- rowSums(as.matrix(dtm_reduced))  # Document lengths
term_frequency <- colSums(as.matrix(dtm_reduced))  # Term frequencies

json_data <- createJSON(
  phi = phi,
  theta = theta,
  vocab = vocab,
  doc.length = doc_length,
  term.frequency = term_frequency
)

serVis(json_data, out.dir = 'LDAvis', open.browser = TRUE)

# Extract necessary data for visualization
phi <- posterior(model_lda_9[["Gibbs"]])$terms  # Term-topic probabilities for Gibbs
theta <- posterior(model_lda_9[["Gibbs"]])$topics  # Document-topic proportions for Gibbs
vocab <- colnames(dtm_reduced)  # Vocabulary terms
doc_length <- rowSums(as.matrix(dtm_reduced))  # Document lengths
term_frequency <- colSums(as.matrix(dtm_reduced))  # Term frequencies

json_data <- createJSON(
  phi = phi,
  theta = theta,
  vocab = vocab,
  doc.length = doc_length,
  term.frequency = term_frequency
)

serVis(json_data, out.dir = 'LDAvis', open.browser = TRUE)

# Extract necessary data for visualization
phi <- posterior(model_lda_10[["Gibbs"]])$terms  # Term-topic probabilities for Gibbs
theta <- posterior(model_lda_10[["Gibbs"]])$topics  # Document-topic proportions for Gibbs
vocab <- colnames(dtm_reduced)  # Vocabulary terms
doc_length <- rowSums(as.matrix(dtm_reduced))  # Document lengths
term_frequency <- colSums(as.matrix(dtm_reduced))  # Term frequencies

json_data <- createJSON(
  phi = phi,
  theta = theta,
  vocab = vocab,
  doc.length = doc_length,
  term.frequency = term_frequency
)

serVis(json_data, out.dir = 'LDAvis', open.browser = TRUE)

#Generating gammaDF table
gammaDF_gibbs<-as.data.frame(model_lda_8[["Gibbs"]]@gamma,
                           row.names=model_lda_8[["Gibbs"]]@documents)

names(gammaDF_gibbs)<-c(1:k) #name the columns with topic id
View(gammaDF_gibbs)