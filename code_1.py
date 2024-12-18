import speech_recognition as sr
import pyttsx3
import time
import threading
import queue

class ContinuousSpeechRecognition:
    def __init__(self, 
                 sample_rate=44100, 
                 chunk_size=1024, 
                 energy_threshold=300, 
                 dynamic_energy_threshold=True):
        """
        Initialize speech recognition with advanced settings
        
        :param sample_rate: Audio sampling rate
        :param chunk_size: Size of audio chunks to process
        :param energy_threshold: Minimum audio energy to detect speech
        :param dynamic_energy_threshold: Automatically adjust energy threshold
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Advanced recognition settings
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        self.recognizer.dynamic_energy_adjustment_ratio = 1.5
        self.recognizer.pause_threshold = 5  # Longer pause before assuming speech ended
        
        # Continuous recording parameters
        self.is_listening = False
        self.text_queue = queue.Queue()
        
        # Output file
        self.output_file = "continuous_output.txt"

    def listen_continuous(self):
        """
        Continuously listen and transcribe speech
        """
        self.is_listening = True
        
        def recognition_thread():
            while self.is_listening:
                try:
                    with self.microphone as source:
                        # Adjust for ambient noise
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        
                        # Extended listening with context
                        print("Listening... (speak now)")
                        audio = self.recognizer.listen(source, 
                                                       phrase_time_limit=None,  # No time limit
                                                       timeout=None)  # No timeout
                        
                        # Transcribe with multiple recognition attempts
                        try:
                            text = self.recognizer.recognize_google(audio, show_all=False)
                            
                            if text:
                                print(f"Transcribed: {text}")
                                self.text_queue.put(text)
                                self.save_text(text)
                        
                        except sr.UnknownValueError:
                            print("Could not understand audio")
                        except sr.RequestError as e:
                            print(f"Could not request results; {e}")
                
                except Exception as e:
                    print(f"Unexpected error in recognition: {e}")
                    time.sleep(1)  # Prevent rapid error loops

        # Start recognition in a separate thread
        recognition_thread = threading.Thread(target=recognition_thread)
        recognition_thread.daemon = True
        recognition_thread.start()

    def save_text(self, text):
        """
        Save transcribed text to file
        
        :param text: Text to save
        """
        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            print(f"Error saving text: {e}")

    def start(self):
        """
        Start continuous speech recognition
        """
        print("Starting continuous speech recognition...")
        self.listen_continuous()

    def stop(self):
        """
        Stop continuous speech recognition
        """
        self.is_listening = False
        print("Stopped speech recognition")

def main():
    # Create speech recognition instance
    speech_rec = ContinuousSpeechRecognition()
    
    try:
        # Start continuous recognition
        speech_rec.start()
        
        # Keep main thread running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        # Allow clean exit
        speech_rec.stop()
        print("\nExiting speech recognition...")

if __name__ == "__main__":
    main()
