import pandas
class Predictor:
    def __init__(self):
        emails = pandas.read_csv("emails.csv")
        emails["words"] = emails["text"].apply(self.process_email)
        self.emails = emails
    def process_email(self, text):
        text = text.lower()
        return list(set(text.split()))
    def calculate_prior_probability(self, word = "lottery"):
        total_emails = len(self.emails)
        spam_emails_slice = self.emails[self.emails["spam"] == 1]
        spam_with_word = spam_emails_slice[spam_emails_slice["words"].apply(lambda words: word in words)]
        return len(spam_with_word) / len(spam_emails_slice)
    def calculate_posterior(self, word = "lottery"):
        total_emails = len(self.emails)
        spam_emails = self.emails[self.emails["spam"] == 1]
        # print(priors)
        # print(priors["lotterySpam"] * priors['spam'], priors['lottery'])
        non_spam_emails = self.emails[self.emails["spam"] == 0]
        spam_with_word = spam_emails[spam_emails["words"].apply(lambda words: word in words)]
        non_spam_with_word = non_spam_emails[non_spam_emails["words"].apply(lambda words: word in words)]
        p_word_given_spam = len(spam_with_word) / len(spam_emails)
        p_word_given_non_spam = len(non_spam_with_word) / len(non_spam_emails)
        
        p_spam = len(spam_emails) / total_emails
        p_non_spam = len(non_spam_emails) / total_emails
        if (p_word_given_spam * p_spam + p_word_given_non_spam * p_non_spam) == 0:
            return 0
        
        p_spam_given_word = (p_word_given_spam * p_spam) / (p_word_given_spam * p_spam + p_word_given_non_spam * p_non_spam)
        return p_spam_given_word

    def predict_naive_bayes(self, word):
        posterior_probability = self.calculate_posterior(word)
        return posterior_probability >= 0.5

word = "lottery1"
instance = Predictor()
print(f"prior probability of '{word}': {instance.calculate_prior_probability(word)}")
print(f"posterir probability of spam given '{word}': {instance.calculate_posterior(word)}")
print(f"Prediction if email is spam given it contains '{word}': {instance.predict_naive_bayes(word)}")