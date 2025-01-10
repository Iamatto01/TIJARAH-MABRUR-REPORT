import wikipediaapi
from gtts import gTTS
from moviepy.editor import TextClip, AudioFileClip, CompositeVideoClip
import os

def get_wikipedia_content(page_title):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(page_title)
    return page.summary if page.exists() else None

def text_to_speech(text, output_file="output.mp3"):
    tts = gTTS(text, lang='en')
    tts.save(output_file)
    return output_file

def create_video_with_audio(text, audio_file, output_file="output_video.mp4"):
    text_clip = TextClip(text, fontsize=40, color='white', size=(1920, 1080), bg_color='black', method='caption')
    text_clip = text_clip.set_duration(30)  # Adjust duration based on text length
    audio = AudioFileClip(audio_file)
    text_clip = text_clip.set_audio(audio)
    text_clip.write_videofile(output_file, fps=24)
    return output_file

# Generate Video
page_title = "Python (programming language)"
content = get_wikipedia_content(page_title)
if content:
    audio_file = text_to_speech(content, "audio.mp3")
    video_file = create_video_with_audio(content, audio_file, "video.mp4")
    print(f"Video created: {video_file}")
else:
    print("Wikipedia page not found!")
