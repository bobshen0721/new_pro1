library(data.table)

iris <-as.data.table(iris)
class(iris$Species)
        
        
model <- lm(Sepal.Width~.,data=iris)
model
