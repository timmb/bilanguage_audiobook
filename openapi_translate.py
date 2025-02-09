import requests
import os
import os.path
import sys

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return api_key

def translate_audio_to_english(file_path, api_key):
    url = "https://api.openai.com/v1/audio/translations"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    base_name, _ = os.path.splitext(file_path)
    translated_text_file = f"{base_name}_translated.txt"
    
    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file, "model": (None, "whisper-1")}
        response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        translated_text = response.json()["text"]
        with open(translated_text_file, "w", encoding="utf-8") as text_file:
            text_file.write(translated_text)
        print(f"Translated text saved to {translated_text_file}")
        return translated_text_file, translated_text
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None, None

def text_to_speech(text, file_path, api_key, voice="onyx", speed=1.0, response_format="mp3"):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    base_name, _ = os.path.splitext(file_path)
    output_audio_path = f"{base_name}_translated.mp3"
    
    data = {
        "model": "tts-1",
        "input": text,
        "voice": voice,
        "response_format": response_format,
        "speed": speed
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        with open(output_audio_path, "wb") as audio_file:
            audio_file.write(response.content)
        print(f"Audio saved to {output_audio_path}")
        return output_audio_path
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

if __name__ == "__main__":
    api_key = get_api_key()
    audio_file_path = "Oscar_Wilde-Privighetoarea_si_trandafirul.mp3"  # Replace with actual file path
    
    translated_text_file, translated_text = translate_audio_to_english(audio_file_path, api_key)
    if translated_text:
        print("Translated Text:", translated_text)
        text_to_speech(translated_text, audio_file_path, api_key)
