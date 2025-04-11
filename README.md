# loanword-matching-models
COGS 402 Project (Jan - April 2025)

This study uses the Serbian Twitter training corpus ReLDI-NormTagNEW-sr 3.0 (Ljubešić et al, 2023), which is a corpus containing 6899 tweets comprised of 92271 word tokens, composed and published by Twitter users in Serbia. This training corpus is a project that began in 2013 and has been iteratively improved and annotated up until 2023 (Ljubešić et al, 2023). To account for recency, this study uses the most updated version of the dataset, containing data points and annotations from 2023. 

To constrain the dataset and minimize manual efforts, three data wrangling methods were implemented, each accounting for a different English loanword adaptation style: 1) raw anglicisms 2) raw acronyms and 3) fuzzy matches (obvious anglicisms). All three methods were trained referencing an English dictionary (Atkinson, 2020) and a Serbian Latinic dictionary (Mihajlović, 2015). It is important to note that written Serbian has two alphabets: Cyrillic and Latinic (Nag, 2017). In this study’s scope, only the Serbian Latinic alphabet is examined. 

