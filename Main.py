import re
import os.path
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


################################################################
#
# TwitterUser class for storing info. about user and his tweets
#
################################################################
class TwitterUser:

    def __init__(self, idd, created, collected, followings,
                 followers, tweets_num, length_name, length_description):
        self._user_id = idd
        self._created_at = created
        self._collected_at = collected
        self._numb_followings = followings
        self._numb_followers = followers
        self._numb_tweets = tweets_num
        self._name_length = length_name
        self._description_length = length_description
        self._tweets = []

        self._tfidf = []
        self._ratio_follower_following = 0.0
        self._count_http = 0
        self._count_at = 0

    @property
    def user_id(self):
        return self._user_id

    @user_id.setter
    def user_id(self, value):
        self._user_id = value

    @property
    def created_at(self):
        return self._created_at

    @created_at.setter
    def created_at(self, value):
        self._created_at = value

    @property
    def collected_at(self):
        return self._collected_at

    @collected_at.setter
    def collected_at(self, value):
        self._collected_at = value

    @property
    def numb_followings(self):
        return self._numb_followings

    @numb_followings.setter
    def numb_followings(self, value):
        self._numb_followings = value

    @property
    def numb_followers(self):
        return self._numb_followers

    @numb_followers.setter
    def numb_followers(self, value):
        self._numb_followers = value

    @property
    def numb_tweets(self):
        return self._numb_tweets

    @numb_tweets.setter
    def numb_tweets(self, value):
        self._numb_tweets = value

    @property
    def name_length(self):
        return self._name_length

    @name_length.setter
    def name_length(self, value):
        self._name_length = value

    @property
    def description_length(self):
        return self._description_length

    @description_length.setter
    def description_length(self, value):
        self._description_length = value

    @property
    def tweets(self):
        return self._tweets

    @tweets.setter
    def tweets(self, value):
        self._tweets = value

    @property
    def ratio_follower_following(self):
        return self._ratio_follower_following

    @ratio_follower_following.setter
    def ratio_follower_following(self, value):
        self._ratio_follower_following = value

    @property
    def count_http(self):
        return self._count_http

    @count_http.setter
    def count_http(self, value):
        self._count_http = value

    @property
    def count_at(self):
        return self._count_at

    @count_at.setter
    def count_at(self, value):
        self._count_at = value

    @property
    def tfidf(self):
        return self._tfidf

    @tfidf.setter
    def tfidf(self, value):
        self._tfidf = value


#################################################################
#
# Imports data from text files, stores it into TwitterUser object
# and finally return a list TwitterUser objects.
#
#################################################################
def import_user_data(user_file, tweet_file):
    twitter_users = []
    with open(user_file) as f1:
        all_user_info = f1.readlines()
        for user_info in all_user_info:
            twitter_user_ob = TwitterUser(re.split(r'\t+', user_info)[0], re.split(r'\t+', user_info)[1],
                                          re.split(r'\t+', user_info)[2], float(re.split(r'\t+', user_info)[3]),
                                          float(re.split(r'\t+', user_info)[4]), float(re.split(r'\t+', user_info)[5]),
                                          float(re.split(r'\t+', user_info)[6]), float(re.split(r'\t+', user_info)[7][:-1]))
            twitter_users.append(twitter_user_ob)

    with open(tweet_file) as f2:
        all_user_tweets = f2.readlines()
        for line_tweets_info in all_user_tweets:
            for twitter_user in twitter_users:
                if twitter_user.user_id == re.split(r'\t+', line_tweets_info)[0]:
                    tweets = twitter_user.tweets
                    tweets.append(re.split(r'\t+', line_tweets_info)[2])
                    twitter_user.tweets = tweets

    return twitter_users


#################################################################
#
# Calculates various features derived from the existing data types
# of the TwitterUser object, and add those calculated features back
# to the object.
#
#################################################################
def calculate_features(twitter_users):
    for user in twitter_users:

        try:
            tfidf = TfidfVectorizer(min_df=1).fit_transform(user.tweets)
            pairwise_similarity = tfidf * tfidf.T
            user.tfidf = csr_matrix.mean(pairwise_similarity).item()
        except Exception:
            user.tfidf = 0.0
            pass

        if user.numb_followings > 0:
            user.ratio_follower_following = user.numb_followers / user.numb_followings
        else:
            user.ratio_follower_following = 0
        at_count = 0
        http_count = 0
        for tweet in user.tweets:
            at_count += tweet.count("@")
            http_count += tweet.count("http")
        user.count_at = at_count
        user.count_http = http_count


####################################################################
#
# Converts the features into numpy arrray / matrix and normalizes it
#
####################################################################
def build_feature_matrix(twitter_users):
    features_matrix = []

    for user in twitter_users:
        features_matrix.append([user.name_length, user.description_length, user.count_http,
                                user.count_at, user.ratio_follower_following, user.tfidf])

    features_matrix_np = np.array(features_matrix)
    features_matrix_normalized = features_matrix_np / features_matrix_np.max(axis=0)

    return features_matrix_normalized


####################################################################
#
# Importing files, creating training & testing features and labels
#
####################################################################
predictUsers = []
def appendToUsers(predictThis):
    for i in predictThis:
	predictUsers.append(i.user_id)



if os.path.isfile("training_labels.dat") and os.path.isfile("training_features_matrix.dat") and os.path.isfile("testing_labels.dat") and os.path.isfile("testing_features_matrix.dat"):
    training_labels = pickle.load(open("training_labels.dat", "r"))
    training_features = pickle.load(open("training_features_matrix.dat", "r"))
    testing_labels = pickle.load(open("testing_labels.dat", "r"))
    testing_features = pickle.load(open("testing_features_matrix.dat", "r"))
    testing_legit = import_user_data("Testing_data/legitimate_users1.txt", "Testing_data/legitimate_users_tweets.txt")
    testing_spammers = import_user_data("Testing_data/spammers1.txt", "Testing_data/spammers_tweets.txt")
    appendToUsers(testing_legit)
    appendToUsers(testing_spammers)
else:
    training_spammers = import_user_data("Training_data/spammers.txt", "Training_data/spammers_tweets.txt")
    calculate_features(training_spammers)
    training_spammers_feature_matrix = build_feature_matrix(training_spammers)

    training_legit = import_user_data("Training_data/legitimate_users.txt", "Training_data/legitimate_users_tweets.txt")
    calculate_features(training_legit)
    training_legit_feature_matrix = build_feature_matrix(training_legit)

    training_labels = [0] * len(training_spammers_feature_matrix) + [1] * len(training_legit_feature_matrix) #0 spammers, 1 - legit
    with open('training_labels.dat', 'w') as outfile:
        pickle.dump(training_labels, outfile)

    training_features = np.concatenate((training_spammers_feature_matrix, training_legit_feature_matrix), axis=0)
    with open('training_features_matrix.dat', 'w') as outfile:
        pickle.dump(training_features, outfile)

    testing_spammers = import_user_data("Testing_data/spammers1.txt", "Testing_data/spammers_tweets.txt")
    calculate_features(testing_spammers)
    testing_spammers_feature_matrix = build_feature_matrix(testing_spammers)
    appendToUsers(testing_spammers)
    testing_legit = import_user_data("Testing_data/legitimate_users1.txt", "Testing_data/legitimate_users_tweets.txt")
    calculate_features(testing_legit)
    testing_legit_feature_matrix = build_feature_matrix(testing_legit)
    appendToUsers(testing_legit)
    testing_labels = [0] * len(testing_spammers_feature_matrix) + [1] * len(testing_legit_feature_matrix)
    with open('testing_labels.dat', 'w') as outfile:
        pickle.dump(testing_labels, outfile)

    testing_features = np.concatenate((testing_spammers_feature_matrix, testing_legit_feature_matrix), axis=0)
    with open('testing_features_matrix.dat', 'w') as outfile:
        pickle.dump(testing_features, outfile)


####################################################################
#
# Configuring plot appearance and labels
#
####################################################################
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.BuPu):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Spammer", "Non-Spammer"], rotation=45)
    plt.yticks(tick_marks, ["Spammer", "Non-Spammer"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####################################################################
#
# Building, testing and evaluating various Machine Learning models.
#
####################################################################
def select_classifier(algo, label):
    model = algo
    model.fit(training_features, training_labels)
    expected = testing_labels
    predicted = model.predict(testing_features)
    count = 0
    for predict in predicted:
	count+=1
	if(predict == 0):
		print(predictUsers[count - 1]),
		print("is a spammer")
	else: 
		print(predictUsers[count - 1]),
		print("is a not spammer")
    print("----------------------------------------------------")
    print("|               Classification Report              |")
    print("----------------------------------------------------")
    print(metrics.classification_report(expected, predicted))
    print("")

    print("----------------------------------------------------")
    print("|                  Confusion Matrix                |")
    print("----------------------------------------------------")
    print(metrics.confusion_matrix(expected, predicted))
    print("")

    cm_list = metrics.confusion_matrix(expected, predicted).tolist()
    list_total = float(sum(sum(x) for x in cm_list))

    print("----------------------------------------------------")
    print("|           False Positives and Negatives          |")
    print("----------------------------------------------------")
    print ("False Positive: ", cm_list[1][0] / list_total)
    print("")
    print ("False Negative: ", cm_list[0][1] / list_total)
    print("")

    plt.figure()
    plot_confusion_matrix(metrics.confusion_matrix(expected, predicted), label)
    plt.show()


# Naive Bias
select_classifier(GaussianNB(), "Naive Bias Classifier")

# SVM
#select_classifier(svm.SVC(), "SVM Classifier")

# ADA Boost
#select_classifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=90), "ADA Boost Classifier")

# Random Forest
#select_classifier(RandomForestClassifier(n_estimators=150), "Random Forest Classifier")





