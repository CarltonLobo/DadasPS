import sounddevice as sd
import numpy as np
import wave
import speech_recognition as sr
import time
from scipy.io import wavfile

def record_audio(filename, duration=10, samplerate=44100):
    """
    Record audio from the default input device (virtual audio cable)
    """
    print(f"Recording audio for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(int(samplerate * duration), 
                      samplerate=samplerate,
                      channels=1,
                      dtype=np.int16)
    sd.wait()
    print("Recording finished!")
    
    # Save the recording as a WAV file
    wavfile.write(filename, samplerate, recording)
    return filename

def convert_speech_to_text(audio_file):
    """
    Convert the recorded audio file to text using speech recognition
    """
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service; {e}"

def main():
    # Get output filename from user
    text_filename = input("Enter the name for the output text file (without extension): ") + ".txt"
    
    # Get recording duration from user
    try:
        duration = float(input("Enter recording duration in seconds: "))
    except ValueError:
        print("Invalid duration. Using default of 10 seconds.")
        duration = 10
    
    # Record audio and save to temporary WAV file
    temp_audio = "temp_recording.wav"
    audio_file = record_audio(temp_audio, duration)
    
    # Convert speech to text
    print("Converting speech to text...")
    text = convert_speech_to_text(audio_file)
    
    # Save text to file
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"\nTranscription saved to {text_filename}")
    print("\nTranscribed text:")
    print("-" * 50)
    print(text)
    print("-" * 50)

if __name__ == "__main__":
    main()