import json
import nltk
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#import sklearn


LOG_FILE="info.log"

def _init():
	nltk.download('averaged_perceptron_tagger')
	sid_obj = SentimentIntensityAnalyzer()
	_init_log_file()	

	return sid_obj

def _init_log_file():
	try:
		f=open(LOG_FILE,'a')
		f.close()
	except FileNotFoundError:
		f=open(LOG_FILE,'w')
		f.close()

def load_json_data(json_file):
	'''
	:param json_file: json file containing speech information

	Example: {"text": "Teste","timestamp":<TIMESTAMP object>}
	'''
	with open(json_file) as fp:	
		d=json.load(fp)	
	return d

def prepare_json_response(data):
	'''
	:param data: response dict with extracted info from speech

	Example: { "Status": Integer,"Entities": [<String>], "Sentiment": String }

	Future iteration: "Emotion Recognized": String
	'''
	with open('response.json','w') as fp:	
		json.dump(data,fp)	


def write_log(info,req=False):	
	if req:
		with open(LOG_FILE,'a') as f:
			if "timestamp" in info.keys():
				f.write(info["timestamp"]+ " REQUEST:"+info["text"]+'\n')
			else:
				f.write(str(datetime.now())+ " REQUEST:"+info["text"]+'\n')
	else:
		with open(LOG_FILE,'a') as f:
			f.write(str(datetime.now()))
			f.write(" RESPONSE:"+str(info)+"\n")				

def _eval_status_response(data):	
	if data['e'] == "" or data['s'] == "":
		return 500
	else:	
		return 200

def process_text(text):
	'''
	tokenize, etc.
	'''
	return text

def extract_entities(text):
	'''
	nltk
	'''
	try:
		#Part-of-Speech
		string=nltk.tokenize.sent_tokenize(text)
		tokens=[nltk.tokenize.word_tokenize(t) for t in string]
		pos=[nltk.pos_tag(token) for token in tokens]

		#Entity-Recognition
	
		all_entities=[]
		for sentence in pos:
			all_entities+=[entity[0] for entity in sentence if 'NN' in entity[1]]
		return all_entities

	except Exception:
		write_log("ERROR processing entity extraction")
		return ""

def sentiment_analysis(model,text):
	'''
	VADER or empath
	'''
	try:
		val=model.polarity_scores(text)
		if val['compound'] >= 0.05:
			return "Positive"
		elif val['compound'] <= -0.05:
			return "Negative"
		else:
			return "Neutral"
	except Exception:
		write_log("ERROR processing sentiment analysis")
		return ""	

def audio_emotion_recognition(audio):

	raise NotImplementedError("Not implemented. Requires: AUDIO_FILE")

if __name__ == '__main__':
	
	model=_init()
	
	'''
	Pipeline
	'''

	data=load_json_data('request.json')

	#write to log timestamp and request info
	write_log(data,True)

	#tokens=process_text(data['text'])

	entities=extract_entities(data['text'])
	sentiment=sentiment_analysis(model,data['text'])

	status=_eval_status_response({"e":entities,"s":sentiment})
	if status==200:
		response={"Status": status, "Entities":entities, "Sentiment": sentiment}
	else:
		response={"Status": status}

	prepare_json_response(response)

	#write to log actual timestamp and response status
	write_log(response)