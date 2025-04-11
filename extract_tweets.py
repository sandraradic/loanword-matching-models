def extract_tweets(conllu_file, output_file):
    """
    Extracts tweets from a .conllu file and saves them to an output file.
    Args:
        conllu_file (str): Path to the .conllu file.
        output_file (str): Path to save the extracted tweets.
    """
    tweets = []
    
    with open(conllu_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("# text ="):
                tweet = line.split("= ", 1)[1].strip()
                tweets.append(tweet)
    
    with open(output_file, "w", encoding="utf-8") as out_file:
        for tweet in tweets:
            out_file.write(tweet + "\n")
    
    print(f"Extracted {len(tweets)} tweets and saved to {output_file}")


if __name__ == "__main__":
    output_path = "output/tweets_all_raw.xlsx"
    intput_path = "input/reldi-normtagner-sr.conllup"
    extract_tweets(intput_path, output_path)

