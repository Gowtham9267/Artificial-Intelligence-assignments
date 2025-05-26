# Required installations:
# pip install SpeechRecognition
# pip install pyaudio

import speech_recognition as sr

def transcribe_audio_speech_recognition(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print("Transcription:", text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

# Example usage
transcribe_audio_speech_recognition("example.wav")
