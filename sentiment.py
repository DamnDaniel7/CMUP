import json
import nltk
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LOG_FILE="info.log"

class Sentiment():
	def __init__(self) -> None:
		self.model = self._initiate()
		self.HOME_CONTEXT=["home","house","room","apartment","kitchen","bedroom","bathroom","aisle"]
		self.DARK_CONTEXT=["dark","black"]
		self.BRIGHT_CONTEXT=["bright","light"]
		self.LOUD_CONTEXT=["loud","noisy","high","sound","big","shout"]
		self.QUIET_CONTEXT=["quiet","low","silence","sound","small","soft","muted"]
		
	def _initiate(self):
		nltk.download('averaged_perceptron_tagger')
		sid_obj = SentimentIntensityAnalyzer()
		self._init_log_file()	

		return sid_obj

	def _init_log_file(self):
		try:
			f=open(LOG_FILE,'a')
			f.close()
		except FileNotFoundError:
			f=open(LOG_FILE,'w')
			f.close()

	def load_json_data(self, json_file):
		'''
		:param json_file: json file containing speech information

		Example: {"text": "Teste","timestamp":<TIMESTAMP object>}
		'''
		with open(json_file) as fp:	
			d=json.load(fp)	
		return d

	def prepare_json_response(self, data):
		'''
		:param data: response dict with extracted info from speech

		Example: { "Status": Integer,"Entities": [<String>], "Sentiment": String }

		Future iteration: "Emotion Recognized": String
		'''
		with open('response.json','w') as fp:	
			json.dump(data,fp)	


	def write_log(self, info, req=False):	
		if req:
			with open(LOG_FILE,'a') as f:
				f.write(str(datetime.now())+ " REQUEST:"+info+'\n')
		else:
			with open(LOG_FILE,'a') as f:
				f.write(str(datetime.now()))
				f.write(" RESPONSE:"+str(info)+"\n")				

	def _eval_status_response(self, data):	
		if data['e'] == "" or data['s'] == "":
			return 500
		else:	
			return 200

	def _extract_relevant_keywords(self,tokens,context):
		'''
		Get relevant keywords from context
		'''
		relevant=[]
		if context=='home':
			relevant+=[token for s in tokens for token in s if token in self.HOME_CONTEXT]
		elif context=='dark':
			relevant+=[token for s in tokens for token in s if token in self.DARK_CONTEXT]
		elif context=='bright':
			relevant+=[token for s in tokens for token in s if token in self.BRIGHT_CONTEXT]
		elif context=='loud':
			relevant+=[token for s in tokens for token in s if token in self.LOUD_CONTEXT]
		elif context=='quiet':
			relevant+=[token for s in tokens for token in s if token in self.QUIET_CONTEXT]
		return relevant

	def extract_entities(self, text,context):
		'''
		nltk
		'''
		try:
			#Part-of-Speech
			string = nltk.tokenize.sent_tokenize(text)
			tokens=[nltk.tokenize.word_tokenize(t) for t in string]
			pos=[nltk.pos_tag(token) for token in tokens]

			relevant_keywords=self._extract_relevant_keywords(tokens,context)			
			#Entity-Recognition
			
			all_entities=[]
			
			for sentence in pos:
				all_entities+=[entity[0] for entity in sentence if 'NN' in entity[1]]
			
			return all_entities+relevant_keywords

		except NotImplementedError:
			self.write_log("ERROR processing entity extraction")
			return ""

	def sentiment_analysis(self, text):
		'''
		VADER or empath
		'''
		try:
			val= self.model.polarity_scores(text)
			if val['compound'] >= 0.05:
				return "Positive"
			elif val['compound'] <= -0.05:
				return "Negative"
			else:
				return "Neutral"
		except Exception:
			self.write_log("ERROR processing sentiment analysis")
			return ""	

	def audio_emotion_recognition(self, audio):
		raise NotImplementedError("Not implemented. Requires: AUDIO_FILE")

	def pipeline(self, texto,context=None):
		'''
		Pipeline
		'''
		#write to log timestamp and request info
		self.write_log(texto,True)

		entities=self.extract_entities(texto,context)
		sentiment=self.sentiment_analysis(texto)

		status=self._eval_status_response({"e":entities,"s":sentiment})
		if status==200:
			response={"Status": status, "Entities":entities, "Sentiment": sentiment}
		else:
			response={"Status": status}

		self.prepare_json_response(response)

		#write to log actual timestamp and response status
		self.write_log(response)

		return response

if __name__ == '__main__':
	sentiment = Sentiment()
	sentiment.pipeline("The TV is too quiet","quiet")