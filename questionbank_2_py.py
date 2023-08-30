import sys
import json
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSimilarityChecker:
    def __init__(self):
        self.dataset = pd.read_csv("C:\\Users\\Subhadeep_Sarkar\\Desktop\\Hackethon\\Final_dataset.csv")
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.dataset['question'].tolist())
        self.similarity_threshold = 0.7
        self.model = LogisticRegression()
        self.model.fit(self.question_vectors, self.dataset['topic'].tolist())

        # Prepare the target variables (marks and weightage)
        self.marks = self.dataset['marks'].values
        self.weightage = self.dataset['weightage'].values

        # Convert marks and weightage to class labels
        self.marks_classes = self.convert_to_classes(self.marks)
        self.weightage_classes = self.convert_to_classes(self.weightage)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_marks_train, self.y_marks_test, self.y_weightage_train, self.y_weightage_test = train_test_split(
            self.question_vectors, self.marks_classes, self.weightage_classes, test_size=0.2, random_state=42
        )

        # Train the Logistic Regression model for marks and weightage
        self.marks_model = LogisticRegression(max_iter=1000)
        self.marks_model.fit(self.X_train, self.y_marks_train)

        self.weightage_model = LogisticRegression(max_iter=1000)
        self.weightage_model.fit(self.X_train, self.y_weightage_train)

    def convert_to_classes(self, values):
        return values

    def check_similarity(self, new_question, given_topic):
        new_question_vector = self.vectorizer.transform([new_question])
        similarities = cosine_similarity(new_question_vector, self.question_vectors)
        similar_indices = [i for i, similarity in enumerate(similarities[0]) if similarity >= self.similarity_threshold]
        similar_questions = self.dataset.iloc[similar_indices]['question'].tolist()
        similar_topics = self.dataset.iloc[similar_indices]['topic'].tolist()

        same_topic_indices = [i for i, topic in enumerate(similar_topics) if topic == given_topic]

        # Check if any similar question exists
        similar_questions_exist = len(similar_questions) > 0

        # Check if any similar question belongs to the same topic
        similar_questions_same_topic = len(same_topic_indices) > 0

        # Check if the topic given by the user is valid
        new_question_vector = self.vectorizer.transform([new_question])
        predicted_topic = self.model.predict(new_question_vector)[0]
        is_valid_topic = given_topic != predicted_topic

        if len(similar_questions) > 0:
            same_question = similar_questions[0]
        else:
            same_question = "null"
        return similar_questions_exist, similar_questions_same_topic, is_valid_topic, same_question

    def predict_marks(self, question_vector):
        predicted_marks_class = self.marks_model.predict(question_vector)
        predicted_marks = self.convert_to_values(predicted_marks_class)[0]
        return predicted_marks

    def predict_weightage(self, question_vector):
        predicted_weightage_class = self.weightage_model.predict(question_vector)
        predicted_weightage = self.convert_to_values(predicted_weightage_class)[0]
        return predicted_weightage

    def convert_to_values(self, classes):
        return classes

if __name__ == "__main__":
    # Usage (same as before)
    similarity_checker = QuestionSimilarityChecker()

    # new_question = "What are the layers of TCP/IP model?"
    # given_topic = "Networking"
    input_data = json.load(sys.stdin)
    new_question = input_data['question']
    given_topic = input_data['topic']

    similar_questions_exist, similar_questions_same_topic, invalid_topic, similar_question = similarity_checker.check_similarity(new_question, given_topic)

    # Transform the new_question into a vector using TfidfVectorizer (same as before)
    new_question_vector = similarity_checker.vectorizer.transform([new_question])

    # Predict the marks and weightage using the trained ML models (Linear Regression)
    marks = similarity_checker.predict_marks(new_question_vector)
    weightage = similarity_checker.predict_weightage(new_question_vector)

    # Prepare the output data as a dictionary (same as before)
    output_data = {
        "similar_questions": similar_questions_exist,
        "same_topic": similar_questions_same_topic,
        "similar_question_name": similar_question,
        "invalid_topic": invalid_topic,
        "marks": int(marks),
        "weightage": int(weightage)
    }

    #print in the csv file
    if output_data["similar_questions"] == False and output_data["same_topic"] == False and output_data["invalid_topic"] == False:
        new_row_data = [int(marks),new_question, "Computer Science", given_topic, int(weightage), "thehitman@gmail.com"]

        # CSV file path
        csv_file_path = "C:\\Users\\Subhadeep_Sarkar\\Desktop\\Hackethon\\Final_dataset.csv"

        # Open the CSV file in write mode
        with open(csv_file_path, mode='a', newline='') as file:
            # Create a csv.writer object
            writer = csv.writer(file)

            # Write the new row to the CSV file
            writer.writerow(new_row_data)

    # Print the output as JSON to stdout (same as before)
    print(json.dumps(output_data))
