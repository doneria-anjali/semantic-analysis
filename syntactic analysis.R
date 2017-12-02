# environment setting
library(dplyr)
library(lattice)
library(ggplot2)
library(caret)
library(proxy)
library(stringr)
library(data.table)
library(e1071)
setwd("./text similarity")

docinp<-read.csv("train.csv")
q1<-docinp$question1
q2<-docinp$question2
d1<-c()
for(i in 1:length(docinp$is_duplicate))
{
  write.table(q1[i], "/Users/Jarvis/Documents/NC State Sem I/ALDA/Project/text similarity/docdata/doc1.txt", sep=" ", row.names = FALSE, col.names = FALSE)
  write.table(q2[i], "/Users/Jarvis/Documents/NC State Sem I/ALDA/Project/text similarity/docdata/doc2.txt", sep=" ", row.names = FALSE, col.names = FALSE)
  # read in original text 
  setwd("/Users/Jarvis/Documents/NC State Sem I/ALDA/Project/text similarity/docdata")
  doc <- lapply( list.files(), readLines)
  
  # preprocess text
  doc1 <- lapply(doc, function(x) {
    text <- gsub("[[:punct:]]", "", x) %>% tolower()
    text <- gsub("\\s+", " ", text) %>% str_trim()  
    word <- strsplit(text, " ") %>% unlist()
    return(word)
  })
  # print only the first text to conserve space
  doc1[[1]]
  ###############################
  
  Shingling <- function(document, k) {
    shingles <- character( length = length(document) - k + 1 )
    
    for( i in 1:( length(document) - k + 1 ) ) {
      shingles[i] <- paste( document[ i:(i + k - 1) ], collapse = " " )
    }
    
    return( unique(shingles) )  
  }
  
  # "shingle" our example document, with k = 3
  doc1 <- lapply(doc1, function(x) {
    Shingling(x, k = 3)
  })
  list( Original = doc[[1]], Shingled = doc1[[1]] )
  
  ###################
  # unique shingles sets across all documents
  doc_dict <- unlist(doc1) %>% unique()
  
  # "characteristic" matrix
  M <- lapply(doc1, function(set, dict) {
    as.integer(dict %in% set)
  }, dict = doc_dict) %>% data.frame() 
  
  # set the names for both rows and columns
  setnames( M, paste( "doc", 1:length(doc1), sep = "_" ) )
  rownames(M) <- doc_dict
  #M
  ###################
  # how similar is two given document, jaccard similarity 
  JaccardSimilarity <- function(x, y) {
    non_zero <- which(x | y)
    set_intersect <- sum( x[non_zero] & y[non_zero] )
    set_union <- length(non_zero)
    return(set_intersect / set_union)
  }
  
  # create a new entry in the registry
  pr_DB$set_entry( FUN = JaccardSimilarity, names = c("JaccardSimilarity") )
  
  # jaccard similarity distance matrix 
  #cluster_similarity(data3$IDS, data3$CESD, similarity="jaccard", method="independence")
  d1[i] <- dist( t(M), method = "JaccardSimilarity" )
  
  # delete the new entry
  pr_DB$delete_entry("JaccardSimilarity")
  d1[i]
}
d1
d2<-round(d1, digits = 0)
d2
confusionMatrix(d2,docinp$is_duplicate)
