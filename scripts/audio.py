import scipy
from diffusers import AudioLDMPipeline

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id)
pipe = pipe.to("cpu")

prompt = "school choir singing american anthem, best quality"
audio = pipe(prompt, num_inference_steps=30, audio_length_in_s=10).audios[0]

# save the audio sample as a .wav file
scipy.io.wavfile.write("anthem.wav", rate=16000, data=audio)