import sounddevice as sd
import numpy as np
import speech_recognition as sr
import wave
import os
import time
import threading

class AdaptiveAudioRecorder:
    def __init__(self, 
                 sample_rate=44100, 
                 channels=1, 
                 pause_threshold=10,  # 10 seconds pause
                 chunk_duration=30):  # 30-second chunks
        """
        Initialize adaptive audio recorder
        
        :param sample_rate: Audio sampling rate
        :param channels: Number of audio channels
        :param pause_threshold: Silence duration to stop recording (in seconds)
        :param chunk_duration: Duration of audio chunks for processing
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.pause_threshold = pause_threshold
        self.chunk_duration = chunk_duration
        
        # Recording parameters
        self.recording = False
        self.frames = []
        self.last_sound_time = None
        
        # Recognizer
        self.recognizer = sr.Recognizer()
        
        # Ensure directories exist
        os.makedirs('recordings', exist_ok=True)
        os.makedirs('transcripts', exist_ok=True)

    def _is_silent(self, audio_chunk, threshold=0.01):
        """
        Check if audio chunk is considered silent
        
        :param audio_chunk: Numpy array of audio data
        :param threshold: Amplitude threshold for silence
        :return: Boolean indicating silence
        """
        return np.abs(audio_chunk).mean() < threshold

    def record_audio(self):
        """
        Continuously record audio with adaptive stopping
        """
        self.recording = True
        self.frames = []
        self.last_sound_time = time.time()

        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            
            # Check if chunk is silent
            if self._is_silent(indata):
                # Check if silence exceeded pause threshold
                if time.time() - self.last_sound_time > self.pause_threshold:
                    self.recording = False
                    return
            else:
                # Update last sound time
                self.last_sound_time = time.time()
            
            # Collect audio frames
            self.frames.append(indata.copy())

        # Start recording stream
        try:
            with sd.InputStream(
                samplerate=self.sample_rate, 
                channels=self.channels,
                callback=audio_callback,
                dtype='float32'
            ):
                # Keep recording until stopped
                while self.recording:
                    sd.sleep(100)  # Small sleep to prevent high CPU usage
        
        except Exception as e:
            print(f"Recording error: {e}")
        finally:
            self.save_recording()

    def save_recording(self):
        """
        Save recorded audio and process transcription
        """
        if not self.frames:
            print("No audio recorded.")
            return

        # Convert frames to numpy array
        audio_data = np.concatenate(self.frames, axis=0)
        
        # Generate unique filename
        timestamp = int(time.time())
        base_filename = f'recordings/recording_{timestamp}'
        
        # Save full recording
        full_wav_path = f'{base_filename}_full.wav'
        self._save_wav(full_wav_path, audio_data)
        
        # Split and process chunks
        self._process_chunks(audio_data, base_filename)

    def _save_wav(self, filename, audio_data):
        """
        Save audio data to WAV file
        
        :param filename: Output filename
        :param audio_data: Numpy array of audio data
        """
        # Scale and convert to int16
        scaled = np.int16(audio_data * 32767)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(scaled.tobytes())
        
        print(f"Saved audio to {filename}")

    def _process_chunks(self, audio_data, base_filename):
        """
        Split audio into chunks and transcribe
        
        :param audio_data: Full audio numpy array
        :param base_filename: Base filename for chunks
        """
        # Calculate chunk size
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        
        # Split audio into chunks
        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i+chunk_samples]
            
            # Skip if chunk is too small
            if len(chunk) < chunk_samples / 2:
                break
            
            # Save chunk
            chunk_filename = f'{base_filename}_chunk_{i//chunk_samples}.wav'
            self._save_wav(chunk_filename, chunk)
            
            # Transcribe chunk
            self._transcribe_chunk(chunk_filename)

    def _transcribe_chunk(self, audio_file):
        """
        Transcribe audio chunk
        
        :param audio_file: Path to audio chunk file
        """
        try:
            # Use SpeechRecognition to transcribe
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                
                try:
                    # Try Google Speech Recognition first
                    text = self.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    print(f"Could not understand audio in {audio_file}")
                    return
                except sr.RequestError:
                    # Fallback to Sphinx if Google fails
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                    except Exception:
                        print(f"Failed to transcribe {audio_file}")
                        return
                
                # Save transcript
                transcript_file = os.path.join(
                    'transcripts', 
                    os.path.basename(audio_file).replace('.wav', '.txt')
                )
                
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"Transcribed {audio_file}: {text[:50]}...")
        
        except Exception as e:
            print(f"Transcription error for {audio_file}: {e}")

def main():
    recorder = AdaptiveAudioRecorder()
    
    print("Audio Recorder Started")
    print("Press Ctrl+C to stop recording")
    
    try:
        recorder.record_audio()
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

if __name__ == "__main__":
    main()