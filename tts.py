from gtts import gTTS
import sys

text  = "안녕?난오디야"
tts = gTTS(text=text, lang='ko')
tts.save("audios/tts.mp3")
